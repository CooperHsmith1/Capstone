"""Common interface for the comparative state-estimation framework.

All three estimators in this package (particle filter, EKF, UKF) estimate the
same 6-D pen-tip state so their outputs are directly comparable:

    x = [px, py, pz, vx, vy, vz]        (env-local frame, metres / m s^-1)

Motion model (shared, linear constant-velocity):

    p_{t+1} = p_t + v_t * dt
    v_{t+1} = v_t                       + process noise ~ N(0, Q)

Measurement model (shared, linear position-only):

    z_t = p_t                           + measurement noise ~ R

NOTE ON THE EKF: because both models above are linear, the EKF's Jacobians are
exactly the constant matrices F and H, so the EKF here reduces algebraically to
a standard linear Kalman filter. That is not a bug and it is not a shortcut --
it is the correct EKF for this system. It matters for the project's research
question: with a *linear* model and *Gaussian* noise all three filters are
near-optimal and should score almost identically. The particle filter only
earns its extra compute once the noise stops being Gaussian (dropouts, outlier
spikes, multimodal ambiguity), which is exactly the regime the benchmark
harness in ``benchmark.py`` injects. Report the equal-performance Gaussian
result as well as the non-Gaussian result -- the crossover point between them
is the interesting finding, not the particle filter winning everywhere.

Every estimator is *batched* over environments: all tensors carry a leading
env dimension B so the filters run inside a vectorised mjlab rollout without a
Python loop over environments.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import torch

# State layout constants -- import these instead of hard-coding indices.
STATE_DIM: int = 6
MEAS_DIM: int = 3
POS_SLICE = slice(0, 3)
VEL_SLICE = slice(3, 6)


@dataclass(kw_only=True)
class StateEstimatorCfg:
  """Shared configuration for every estimator in the comparison.

  Using one config class for all three filters is deliberate: it guarantees
  the benchmark compares filters under *identical* noise assumptions, so any
  performance difference is attributable to the algorithm rather than to
  mismatched tuning.

  Attributes:
      process_noise_pos: Process-noise std dev on position (m). Small --
          position is driven by velocity, not by direct disturbance.
      process_noise_vel: Process-noise std dev on velocity (m/s). This is the
          main tuning knob: it encodes how much the constant-velocity
          assumption is violated by the arm's real acceleration. Too small
          and the filter lags the arm; too large and it just tracks noise.
      measurement_noise: Measurement-noise std dev on pen position (m).
          Should match the noise actually injected into the observation.
      initial_pos_std: Std dev of the initial position spread at reset (m).
      initial_vel_std: Std dev of the initial velocity spread at reset (m/s).
  """

  process_noise_pos: float = 0.001
  process_noise_vel: float = 0.15
  measurement_noise: float = 0.01
  initial_pos_std: float = 0.05
  initial_vel_std: float = 0.10

  def process_noise_cov(self, device: torch.device) -> torch.Tensor:
    """Return Q as a [6, 6] diagonal covariance matrix."""
    diag = torch.tensor(
      [self.process_noise_pos**2] * 3 + [self.process_noise_vel**2] * 3,
      device=device,
      dtype=torch.float32,
    )
    return torch.diag(diag)

  def measurement_noise_cov(self, device: torch.device) -> torch.Tensor:
    """Return R as a [3, 3] diagonal covariance matrix."""
    diag = torch.full(
      (MEAS_DIM,),
      self.measurement_noise**2,
      device=device,
      dtype=torch.float32,
    )
    return torch.diag(diag)


def constant_velocity_transition(dt: float, device: torch.device) -> torch.Tensor:
  """Return the [6, 6] state-transition matrix F for a timestep of ``dt``.

  F = [[I, dt*I],
       [0,    I]]
  """
  F = torch.eye(STATE_DIM, device=device, dtype=torch.float32)
  F[POS_SLICE, VEL_SLICE] = torch.eye(3, device=device, dtype=torch.float32) * dt
  return F


def position_measurement_matrix(device: torch.device) -> torch.Tensor:
  """Return the [3, 6] measurement matrix H that selects position from state."""
  H = torch.zeros(MEAS_DIM, STATE_DIM, device=device, dtype=torch.float32)
  H[:, POS_SLICE] = torch.eye(3, device=device, dtype=torch.float32)
  return H


class StateEstimator(abc.ABC):
  """Abstract batched recursive state estimator.

  Subclasses implement ``_predict``, ``_update`` and ``_reset_idx``; the
  public ``predict`` / ``update`` / ``reset`` wrappers handle shape checking
  and diagnostics so every filter reports the same metrics.

  Args:
      num_envs: Batch size B (number of parallel environments).
      cfg: Shared noise configuration.
      device: Torch device the filter runs on.
  """

  def __init__(
    self,
    num_envs: int,
    cfg: StateEstimatorCfg,
    device: torch.device | str = "cpu",
  ) -> None:
    self.num_envs = int(num_envs)
    self.cfg = cfg
    self.device = torch.device(device)

    self._Q = cfg.process_noise_cov(self.device)
    self._R = cfg.measurement_noise_cov(self.device)
    self._H = position_measurement_matrix(self.device)

    # Diagnostics, updated every step. Kept per-env (shape [B]) so they can
    # be logged per-environment or reduced by the caller.
    self._last_innovation = torch.zeros(self.num_envs, MEAS_DIM, device=self.device)

  # ------------------------------------------------------------------
  # Public API
  # ------------------------------------------------------------------

  @property
  @abc.abstractmethod
  def name(self) -> str:
    """Short identifier used in benchmark tables and logs."""

  @property
  @abc.abstractmethod
  def state(self) -> torch.Tensor:
    """Current state estimate, shape [B, 6]."""

  @property
  @abc.abstractmethod
  def covariance(self) -> torch.Tensor:
    """Current state covariance, shape [B, 6, 6]."""

  @property
  def position(self) -> torch.Tensor:
    """Estimated pen-tip position, shape [B, 3]."""
    return self.state[:, POS_SLICE]

  @property
  def velocity(self) -> torch.Tensor:
    """Estimated pen-tip velocity, shape [B, 3]."""
    return self.state[:, VEL_SLICE]

  @property
  def position_std(self) -> torch.Tensor:
    """Per-axis position standard deviation, shape [B, 3].

    Exposed as an observation so the policy can see *how confident* the
    estimator is, not just what it estimates -- this is what lets a policy
    learn to slow down when localisation degrades.
    """
    var = torch.diagonal(self.covariance, dim1=-2, dim2=-1)[:, POS_SLICE]
    return torch.sqrt(torch.clamp(var, min=0.0))

  @property
  def last_innovation(self) -> torch.Tensor:
    """Most recent measurement residual (z - H x), shape [B, 3].

    A persistently large innovation means the filter has diverged from the
    measurements -- useful as a health check during training.
    """
    return self._last_innovation

  def predict(self, dt: float) -> None:
    """Advance the estimate one timestep through the motion model."""
    self._predict(dt)

  def update(self, measurement: torch.Tensor) -> None:
    """Correct the estimate with a noisy pen-position measurement.

    Args:
        measurement: Observed pen-tip position, shape [B, 3].
    """
    if measurement.shape != (self.num_envs, MEAS_DIM):
      raise ValueError(
        f"{self.name}.update expected measurement of shape "
        f"({self.num_envs}, {MEAS_DIM}), got {tuple(measurement.shape)}."
      )
    measurement = measurement.to(self.device, dtype=torch.float32)
    self._last_innovation = measurement - self.position
    self._update(measurement)

  def reset(
    self,
    env_ids: torch.Tensor | None = None,
    measurement: torch.Tensor | None = None,
  ) -> None:
    """Reinitialise the estimator for the given environments.

    Args:
        env_ids: Indices to reset, shape [K]. ``None`` resets all envs.
        measurement: Optional pen positions to centre the new estimate on,
            shape [K, 3] (or [B, 3] when ``env_ids`` is None). Seeding from
            the first real measurement converges far faster than seeding
            from zeros, which starts the filter ~0.6 m from the board.
    """
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    env_ids = env_ids.to(self.device)
    if env_ids.numel() == 0:
      return
    if measurement is not None:
      measurement = measurement.to(self.device, dtype=torch.float32)
      if measurement.shape[0] != env_ids.shape[0]:
        raise ValueError(
          f"{self.name}.reset: measurement has "
          f"{measurement.shape[0]} rows but env_ids has "
          f"{env_ids.shape[0]}."
        )
    self._last_innovation[env_ids] = 0.0
    self._reset_idx(env_ids, measurement)

  def step(self, dt: float, measurement: torch.Tensor) -> torch.Tensor:
    """Convenience: one full predict -> update cycle.

    Returns:
        The posterior position estimate, shape [B, 3].
    """
    self.predict(dt)
    self.update(measurement)
    return self.position

  # ------------------------------------------------------------------
  # Subclass hooks
  # ------------------------------------------------------------------

  @abc.abstractmethod
  def _predict(self, dt: float) -> None: ...

  @abc.abstractmethod
  def _update(self, measurement: torch.Tensor) -> None: ...

  @abc.abstractmethod
  def _reset_idx(
    self, env_ids: torch.Tensor, measurement: torch.Tensor | None
  ) -> None: ...
