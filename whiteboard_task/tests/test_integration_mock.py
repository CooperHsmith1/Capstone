"""End-to-end test of the estimator integration against a mock mjlab env.

Syntax-checking ``integration.py`` and ``observations.py`` proves nothing about
whether they actually work, and mjlab is not installed here. This module stands
up a minimal fake of the mjlab surface the drawing task touches (Entity,
SceneEntityCfg, scene indexing, env_origins, step counter) and drives the real
integration code through it.

It catches the wiring bugs that matter: wrong tensor frames, the estimator
being stepped twice per env step or not at all, resets not propagating, and
observation terms disagreeing about which filter instance they are reading.

Run with:  python tests/test_integration_mock.py
"""

from __future__ import annotations

import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Minimal mjlab stand-in, installed into sys.modules before importing our code
# ---------------------------------------------------------------------------


def install_mock_mjlab() -> None:
  """Register fake mjlab modules so the real integration code can import."""

  class Entity:
    def __init__(self, num_envs: int, num_sites: int = 4):
      self.data = types.SimpleNamespace(
        site_pos_w=torch.zeros(num_envs, num_sites, 3),
        joint_vel=torch.zeros(num_envs, 7),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]] * num_envs),
      )

  class SceneEntityCfg:
    def __init__(self, name, site_names=None, body_names=None, joint_names=None):
      self.name = name
      self.site_names = site_names
      self.body_names = body_names
      self.joint_names = joint_names
      self.site_ids: list[int] | slice = [0]
      self.body_ids: list[int] | slice = [0]
      self.joint_ids: list[int] | slice = slice(None)

    def resolve(self, scene):
      # Pretend "pen_tip" resolves to site index 2.
      if self.site_names and "pen_tip" in self.site_names:
        self.site_ids = [2]

  entity_mod = types.ModuleType("mjlab.entity")
  entity_mod.Entity = Entity

  sec_mod = types.ModuleType("mjlab.managers.scene_entity_config")
  sec_mod.SceneEntityCfg = SceneEntityCfg

  sensor_mod = types.ModuleType("mjlab.sensor")
  sensor_mod.ContactSensor = object
  ths_mod = types.ModuleType("mjlab.sensor.terrain_height_sensor")
  ths_mod.TerrainHeightSensor = object

  envs_mod = types.ModuleType("mjlab.envs")
  envs_mod.ManagerBasedRlEnv = object

  class CommandTerm:
    def __init__(self, cfg, env):
      self.cfg = cfg
      self._env = env
      self.num_envs = env.num_envs
      self.device = env.device
      self.metrics: dict = {}

  class CommandTermCfg:
    pass

  cmd_mod = types.ModuleType("mjlab.managers.command_manager")
  cmd_mod.CommandTerm = CommandTerm
  cmd_mod.CommandTermCfg = CommandTermCfg

  mjlab = types.ModuleType("mjlab")
  mjlab.__path__ = []  # mark as a package
  managers = types.ModuleType("mjlab.managers")
  managers.__path__ = []  # mark as a package
  sensor_mod.__path__ = []  # has a submodule too

  for name, mod in [
    ("mjlab", mjlab),
    ("mjlab.entity", entity_mod),
    ("mjlab.managers", managers),
    ("mjlab.managers.scene_entity_config", sec_mod),
    ("mjlab.managers.command_manager", cmd_mod),
    ("mjlab.sensor", sensor_mod),
    ("mjlab.sensor.terrain_height_sensor", ths_mod),
    ("mjlab.envs", envs_mod),
  ]:
    sys.modules[name] = mod

  return Entity


Entity = install_mock_mjlab()

from mdp.observations import (  # noqa: E402
  estimated_pen_position,
  estimated_pen_velocity,
  pen_position_uncertainty,
)
from mdp.state_estimation.integration import (  # noqa: E402
  StateEstimationCfg,
  get_state_estimator,
  reset_state_estimator,
)


class MockScene:
  def __init__(self, num_envs: int):
    self._robot = Entity(num_envs)
    # Tile envs 5 m apart in X so the env-local frame conversion is
    # actually exercised -- if env_origins is not subtracted somewhere,
    # the error explodes and the test fails loudly.
    self.env_origins = torch.zeros(num_envs, 3)
    self.env_origins[:, 0] = torch.arange(num_envs).float() * 5.0

  def __getitem__(self, key):
    assert key == "robot", f"unexpected scene key {key!r}"
    return self._robot


class MockEnv:
  """Just enough of ManagerBasedRlEnv for the estimator to run."""

  def __init__(self, num_envs: int = 4, dt: float = 0.02):
    self.num_envs = num_envs
    self.device = "cpu"
    self.step_dt = dt
    self.scene = MockScene(num_envs)
    self.common_step_counter = 0
    self.cfg = types.SimpleNamespace(
      sim=types.SimpleNamespace(mujoco=types.SimpleNamespace(timestep=0.005)),
      decimation=4,
    )

  def set_pen_world(self, local_pos: torch.Tensor) -> None:
    """Place the pen at an env-local position, stored as world coords."""
    self.scene._robot.data.site_pos_w[:, 2, :] = local_pos + self.scene.env_origins

  def advance(self) -> None:
    self.common_step_counter += 1


def trajectory(step: int, num_envs: int, dt: float = 0.02) -> torch.Tensor:
  """Env-local pen position at a given step, shape [B, 3]."""
  t = step * dt
  phase = torch.linspace(0.0, 1.0, num_envs)
  x = torch.full((num_envs,), 0.63)
  y = 0.25 * torch.sin(1.5 * t + phase)
  z = 1.10 + 0.20 * torch.sin(2.3 * t + phase)
  return torch.stack([x, y, z], dim=-1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_estimator_tracks_in_env_local_frame():
  """The estimate must be env-local, not world, despite tiled env origins."""
  env = MockEnv(num_envs=4)
  cfg = StateEstimationCfg(filter_type="particle", sensor_noise_std=0.01)

  est_errs = []
  for step in range(300):
    truth = trajectory(step, env.num_envs)
    env.set_pen_world(truth)
    env.advance()
    est = estimated_pen_position(env, cfg)
    if step > 20:  # allow the filter to converge
      est_errs.append(torch.linalg.norm(est - truth, dim=-1))

  err = torch.stack(est_errs).mean()
  assert err < 0.05, (
    f"Estimate is {err:.3f} m from truth -- if this is ~5 m or more, "
    "env_origins is not being subtracted somewhere."
  )
  print(f"PASS  env-local tracking (mean error {err * 1000:.2f} mm)")


def test_estimator_beats_raw_sensor_in_env():
  """Through the full integration path, the filter must reduce sensor noise."""
  env = MockEnv(num_envs=4)
  cfg = StateEstimationCfg(filter_type="particle", sensor_noise_std=0.02)
  manager = get_state_estimator(env, cfg)

  for step in range(400):
    env.set_pen_world(trajectory(step, env.num_envs))
    env.advance()
    estimated_pen_position(env, cfg)

  filtered = float(manager.rmse().mean())
  raw = float(manager.raw_rmse().mean())
  assert filtered < raw, f"filter RMSE {filtered:.4f} >= raw {raw:.4f}"
  print(
    f"PASS  noise reduction through env "
    f"({raw * 1000:.2f} mm raw -> {filtered * 1000:.2f} mm filtered)"
  )


def test_stepped_once_per_env_step():
  """Multiple observation terms in one step must not advance the filter twice.

  If each term stepped the filter independently, three terms would triple the
  filter's effective timestep and the estimate would run ahead of the pen.
  """
  env = MockEnv(num_envs=2)
  cfg = StateEstimationCfg(filter_type="ekf", sensor_noise_std=0.0)
  manager = get_state_estimator(env, cfg)

  env.set_pen_world(trajectory(0, env.num_envs))
  env.advance()

  # Call all three estimator-backed observation terms in one step.
  estimated_pen_position(env, cfg)
  count_after_first = manager._sq_err_count.clone()
  estimated_pen_velocity(env, cfg)
  pen_position_uncertainty(env, cfg)
  count_after_all = manager._sq_err_count

  assert torch.equal(count_after_first, count_after_all), (
    f"filter advanced {count_after_all[0]:.0f} times in one env step, "
    f"expected {count_after_first[0]:.0f}"
  )
  assert float(count_after_all[0]) == 1.0
  print("PASS  filter advances exactly once per env step")


def test_reset_reseeds_filter():
  """After a reset event, the filter must re-seed rather than chase a jump."""
  env = MockEnv(num_envs=4)
  cfg = StateEstimationCfg(filter_type="particle", sensor_noise_std=0.005)

  for step in range(100):
    env.set_pen_world(trajectory(step, env.num_envs))
    env.advance()
    estimated_pen_position(env, cfg)

  # Teleport the pen (as an episode reset would) and reset two envs.
  jumped = trajectory(0, env.num_envs) + torch.tensor([0.0, -0.30, 0.30])
  env.set_pen_world(jumped)
  reset_state_estimator(env, torch.tensor([0, 2]))
  env.advance()
  est = estimated_pen_position(env, cfg)

  reset_err = torch.linalg.norm(est[[0, 2]] - jumped[[0, 2]], dim=-1).mean()
  stale_err = torch.linalg.norm(est[[1, 3]] - jumped[[1, 3]], dim=-1).mean()

  assert reset_err < stale_err, (
    f"reset envs ({reset_err:.4f} m) should be closer to the new position "
    f"than un-reset envs ({stale_err:.4f} m)"
  )
  print(
    f"PASS  reset re-seeds ({reset_err * 1000:.1f} mm reset vs "
    f"{stale_err * 1000:.1f} mm stale)"
  )


def test_all_filter_types_run_in_env():
  """Every filter type must work through the integration path."""
  for ftype in ("particle", "ekf", "ukf"):
    env = MockEnv(num_envs=3)
    cfg = StateEstimationCfg(filter_type=ftype, sensor_noise_std=0.01)
    for step in range(60):
      env.set_pen_world(trajectory(step, env.num_envs))
      env.advance()
      out = estimated_pen_position(env, cfg)
    assert out.shape == (3, 3)
    assert torch.isfinite(out).all(), f"{ftype} produced non-finite output"
  print("PASS  all filter types run in env (particle, ekf, ukf)")


def test_observation_shapes():
  """Observation terms must return the shapes the obs manager expects."""
  env = MockEnv(num_envs=5)
  cfg = StateEstimationCfg(filter_type="particle")
  env.set_pen_world(trajectory(0, env.num_envs))
  env.advance()
  assert estimated_pen_position(env, cfg).shape == (5, 3)
  assert estimated_pen_velocity(env, cfg).shape == (5, 3)
  assert pen_position_uncertainty(env, cfg).shape == (5, 3)
  assert (pen_position_uncertainty(env, cfg) >= 0).all(), "std must be non-negative"
  print("PASS  observation term shapes")


def test_metrics_available():
  """Logging metrics must be finite and sensibly signed."""
  env = MockEnv(num_envs=4)
  cfg = StateEstimationCfg(filter_type="particle", sensor_noise_std=0.02)
  manager = get_state_estimator(env, cfg)
  for step in range(200):
    env.set_pen_world(trajectory(step, env.num_envs))
    env.advance()
    estimated_pen_position(env, cfg)

  m = manager.metrics()
  for key, val in m.items():
    assert val == val and abs(val) < 1e6, f"{key} is not a sane value: {val}"
  assert m["state_est/improvement"] > 0.0, "filter is not improving on the sensor"
  print(
    "PASS  metrics  " + "  ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in m.items())
  )


if __name__ == "__main__":
  torch.manual_seed(0)
  test_estimator_tracks_in_env_local_frame()
  test_estimator_beats_raw_sensor_in_env()
  test_stepped_once_per_env_step()
  test_reset_reseeds_filter()
  test_all_filter_types_run_in_env()
  test_observation_shapes()
  test_metrics_available()
  print("\nAll integration tests passed.")
