"""Wiring the state estimators into the mjlab simulation loop.

Design note -- why the estimator is driven from the observation manager rather
than from a post-physics hook:

mjlab does not expose a stable public "after physics, before observations"
callback, and the private attribute paths differ between versions (the same
problem ``drawing_env_cfg._get_mj_model`` works around). Monkey-patching a
private method is fragile and breaks silently on upgrade.

Instead the estimator is advanced lazily from inside the observation terms,
guarded by the environment's step counter so it advances exactly once per
control step no matter how many observation terms ask for it. That gives the
correct predict -> update ordering relative to physics without depending on any
mjlab internals, and it degrades safely: if an observation term is removed from
the config, the estimator simply stops being stepped rather than crashing.

The trade-off is that the estimator only runs when at least one estimator-backed
observation term is active. That is the behaviour you want -- it costs nothing
when the feature is switched off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ..cfg_utils import resolve_first_site_id
from .base import StateEstimator, StateEstimatorCfg
from .kalman import ExtendedKalmanFilter, UnscentedKalmanFilter
from .particle_filter import ParticleFilter

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

FilterType = Literal["particle", "ekf", "ukf"]

# Attribute name used to cache the manager on the env instance.
_ATTR = "_whiteboard_state_estimator"


@dataclass(kw_only=True)
class StateEstimationCfg:
  """Configuration for the environment-attached state estimator.

  Attributes:
      filter_type: Which estimator to run: ``"particle"``, ``"ekf"`` or
          ``"ukf"``.
      noise: Shared process/measurement noise configuration.
      num_particles: Ensemble size (particle filter only).
      proposal: ``"optimal"`` or ``"bootstrap"`` (particle filter only).
      resample_threshold: ESS fraction below which to resample.
      sensor_noise_std: Std dev of the synthetic sensor noise injected into
          the *true* pen position to create the measurement the filter sees,
          in metres. This is the domain-randomisation knob from Section 5.2 of
          the report -- it is what makes the estimator do real work in sim
          instead of filtering a perfect signal.
      outlier_rate: Probability per step that a measurement is replaced by a
          gross outlier. Models dropouts and tracking glitches; this is the
          regime where the particle filter earns its cost.
      outlier_scale: Outlier magnitude as a multiple of ``sensor_noise_std``.
      asset_cfg: Entity/site config resolving the pen tip.
  """

  filter_type: FilterType = "particle"
  noise: StateEstimatorCfg = field(default_factory=StateEstimatorCfg)
  num_particles: int = 512
  proposal: str = "optimal"
  resample_threshold: float = 0.5
  sensor_noise_std: float = 0.01
  outlier_rate: float = 0.0
  outlier_scale: float = 25.0
  asset_cfg: SceneEntityCfg = field(
    default_factory=lambda: SceneEntityCfg("robot", site_names=("pen_tip",))
  )


def build_estimator(
  cfg: StateEstimationCfg, num_envs: int, device: torch.device | str
) -> StateEstimator:
  """Instantiate the estimator named by ``cfg.filter_type``."""
  if cfg.filter_type == "particle":
    return ParticleFilter(
      num_envs,
      cfg.noise,
      num_particles=cfg.num_particles,
      resample_threshold=cfg.resample_threshold,
      proposal=cfg.proposal,
      device=device,
    )
  if cfg.filter_type == "ekf":
    return ExtendedKalmanFilter(num_envs, cfg.noise, device=device)
  if cfg.filter_type == "ukf":
    return UnscentedKalmanFilter(num_envs, cfg.noise, device=device)
  raise ValueError(
    f"Unknown filter_type {cfg.filter_type!r}; expected 'particle', 'ekf' or 'ukf'."
  )


class StateEstimatorManager:
  """Owns one estimator and advances it once per environment step.

  Args:
      env: The mjlab environment.
      cfg: Estimation configuration.
  """

  def __init__(self, env: ManagerBasedRlEnv, cfg: StateEstimationCfg) -> None:
    self._env = env
    self.cfg = cfg
    self.device = torch.device(getattr(env, "device", "cpu"))
    self.num_envs = int(getattr(env, "num_envs", 1))

    cfg.asset_cfg.resolve(env.scene)
    self._site_idx = resolve_first_site_id(cfg.asset_cfg)

    self.estimator = build_estimator(cfg, self.num_envs, self.device)

    self._last_step: int = -1
    self._initialised = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # Running diagnostics against ground truth. Only available in sim, but
    # that is exactly where the benchmarking happens.
    self._sq_err_sum = torch.zeros(self.num_envs, device=self.device)
    self._sq_err_count = torch.zeros(self.num_envs, device=self.device)
    self._raw_sq_err_sum = torch.zeros(self.num_envs, device=self.device)

  # ------------------------------------------------------------------
  # Environment plumbing
  # ------------------------------------------------------------------

  @property
  def step_dt(self) -> float:
    """Control timestep in seconds (physics timestep x decimation)."""
    dt = getattr(self._env, "step_dt", None)
    if dt is not None:
      return float(dt)
    cfg = self._env.cfg
    return float(cfg.sim.mujoco.timestep) * float(cfg.decimation)

  def _step_counter(self) -> int:
    """Read the env's global step counter, with a local fallback."""
    for attr in ("common_step_counter", "episode_length_buf", "_step_count"):
      val = getattr(self._env, attr, None)
      if val is None:
        continue
      if isinstance(val, torch.Tensor):
        return int(val.max().item())
      return int(val)
    return self._last_step + 1

  def true_pen_position(self) -> torch.Tensor:
    """Ground-truth env-local pen-tip position, shape [B, 3]."""
    asset: Entity = self._env.scene[self.cfg.asset_cfg.name]
    pos_w = asset.data.site_pos_w[:, self._site_idx, :]
    return pos_w - self._env.scene.env_origins

  def _measure(self, truth: torch.Tensor) -> torch.Tensor:
    """Corrupt the true position into a synthetic sensor reading."""
    noisy = truth + torch.randn_like(truth) * self.cfg.sensor_noise_std
    if self.cfg.outlier_rate > 0.0:
      mask = torch.rand(self.num_envs, device=self.device) < self.cfg.outlier_rate
      spike = torch.randn_like(truth) * (
        self.cfg.sensor_noise_std * self.cfg.outlier_scale
      )
      noisy = torch.where(mask.unsqueeze(-1), truth + spike, noisy)
    return noisy

  # ------------------------------------------------------------------
  # Stepping
  # ------------------------------------------------------------------

  def step_if_needed(self) -> None:
    """Advance the filter one control step, at most once per env step."""
    step = self._step_counter()
    if step == self._last_step:
      return
    self._last_step = step

    truth = self.true_pen_position()
    measurement = self._measure(truth)

    # Seed any environment that has not been initialised yet (first step of
    # the run, or the step after a reset) directly from its measurement.
    fresh = ~self._initialised
    if bool(fresh.any()):
      ids = torch.nonzero(fresh, as_tuple=False).squeeze(-1)
      self.estimator.reset(env_ids=ids, measurement=measurement[ids])
      self._initialised[ids] = True

    self.estimator.predict(self.step_dt)
    self.estimator.update(measurement)

    # Diagnostics.
    err = torch.sum((self.estimator.position - truth) ** 2, dim=-1)
    raw_err = torch.sum((measurement - truth) ** 2, dim=-1)
    self._sq_err_sum += err
    self._raw_sq_err_sum += raw_err
    self._sq_err_count += 1.0

  def reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
    """Mark environments for re-seeding on their next step."""
    if env_ids is None:
      self._initialised[:] = False
      self._sq_err_sum[:] = 0.0
      self._raw_sq_err_sum[:] = 0.0
      self._sq_err_count[:] = 0.0
    else:
      self._initialised[env_ids] = False
      self._sq_err_sum[env_ids] = 0.0
      self._raw_sq_err_sum[env_ids] = 0.0
      self._sq_err_count[env_ids] = 0.0

  # ------------------------------------------------------------------
  # Metrics
  # ------------------------------------------------------------------

  def rmse(self) -> torch.Tensor:
    """Running per-env RMSE of the estimate against truth, shape [B]."""
    count = torch.clamp(self._sq_err_count, min=1.0)
    return torch.sqrt(self._sq_err_sum / count)

  def raw_rmse(self) -> torch.Tensor:
    """Running per-env RMSE of the raw measurement, shape [B].

    Compare against ``rmse()``: if the filter is not beating this, it is
    adding latency for nothing.
    """
    count = torch.clamp(self._sq_err_count, min=1.0)
    return torch.sqrt(self._raw_sq_err_sum / count)

  def metrics(self) -> dict[str, float]:
    """Scalar summary suitable for logging to TensorBoard / W&B."""
    return {
      "state_est/rmse_m": float(self.rmse().mean()),
      "state_est/raw_rmse_m": float(self.raw_rmse().mean()),
      "state_est/improvement": float(
        1.0 - (self.rmse().mean() / torch.clamp(self.raw_rmse().mean(), min=1e-9))
      ),
      "state_est/innovation_norm_m": float(
        self.estimator.last_innovation.norm(dim=-1).mean()
      ),
      "state_est/pos_std_m": float(self.estimator.position_std.mean()),
    }


# ---------------------------------------------------------------------------
# Accessor used by observation / reward terms
# ---------------------------------------------------------------------------


def get_state_estimator(
  env: ManagerBasedRlEnv,
  cfg: StateEstimationCfg | None = None,
) -> StateEstimatorManager:
  """Return the env's estimator manager, creating it on first use.

  The manager is cached on the env instance, so every observation term that
  calls this shares one filter rather than each building its own.
  """
  manager = getattr(env, _ATTR, None)
  if manager is None:
    manager = StateEstimatorManager(env, cfg or StateEstimationCfg())
    setattr(env, _ATTR, manager)
  return manager


def has_state_estimator(env: ManagerBasedRlEnv) -> bool:
  """Return True if an estimator has been attached to this env."""
  return getattr(env, _ATTR, None) is not None


def reset_state_estimator(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
) -> None:
  """Event term: re-seed the estimator for the environments being reset.

  Register with ``mode="reset"``::

      events["reset_state_estimator"] = EventTermCfg(
          func=reset_state_estimator,
          mode="reset",
          params={},
      )

  Without this the filter carries its pre-reset belief across the episode
  boundary and spends the first several steps of every new episode chasing a
  pen that teleported. It is a no-op if no estimator has been attached, so it
  is safe to leave registered when the feature is switched off.
  """
  manager = getattr(env, _ATTR, None)
  if manager is not None:
    manager.reset_idx(env_ids)
