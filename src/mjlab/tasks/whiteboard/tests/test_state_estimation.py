"""Standalone tests for the state-estimation package.

These deliberately import nothing from mjlab so they run on any machine with
torch installed -- including CI and your home training box before the sim is
even built. Run with:

    python -m pytest tests/test_state_estimation.py -v

or directly:

    python tests/test_state_estimation.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mdp.state_estimation.base import StateEstimatorCfg  # noqa: E402
from mdp.state_estimation.kalman import (  # noqa: E402
  ExtendedKalmanFilter,
  UnscentedKalmanFilter,
)
from mdp.state_estimation.particle_filter import ParticleFilter  # noqa: E402


def make_trajectory(
  steps: int = 400,
  num_envs: int = 4,
  dt: float = 0.02,
  device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
  """Generate a smooth pen trajectory sweeping across the whiteboard face.

  Returns:
      positions: [T, B, 3] ground-truth pen positions.
      velocities: [T, B, 3] ground-truth pen velocities.
  """
  t = torch.arange(steps, device=device, dtype=torch.float32) * dt
  t = t.unsqueeze(1).expand(steps, num_envs)

  phase = torch.linspace(0.0, 1.5, num_envs, device=device).unsqueeze(0)

  # X pinned near the board face, Y/Z tracing a Lissajous figure.
  x = torch.full_like(t, 0.63) + 0.005 * torch.sin(4.0 * t + phase)
  y = 0.25 * torch.sin(1.5 * t + phase)
  z = 1.10 + 0.20 * torch.sin(2.3 * t + phase)
  positions = torch.stack([x, y, z], dim=-1)

  velocities = torch.zeros_like(positions)
  velocities[1:] = (positions[1:] - positions[:-1]) / dt
  return positions, velocities


def add_noise(
  positions: torch.Tensor,
  sigma: float,
  outlier_rate: float = 0.0,
  outlier_scale: float = 25.0,
  seed: int = 0,
) -> torch.Tensor:
  """Corrupt ground truth with Gaussian noise plus optional outlier spikes."""
  g = torch.Generator(device=positions.device)
  g.manual_seed(seed)
  noisy = (
    positions
    + torch.randn(positions.shape, generator=g, device=positions.device) * sigma
  )
  if outlier_rate > 0.0:
    mask = (
      torch.rand(positions.shape[:2], generator=g, device=positions.device)
      < outlier_rate
    )
    spike = torch.randn(positions.shape, generator=g, device=positions.device) * (
      sigma * outlier_scale
    )
    noisy = torch.where(mask.unsqueeze(-1), positions + spike, noisy)
  return noisy


def run_filter(estimator, measurements, dt) -> torch.Tensor:
  """Run a filter over a measurement sequence. Returns [T, B, 3] estimates."""
  estimator.reset(measurement=measurements[0])
  out = []
  for t in range(measurements.shape[0]):
    estimator.predict(dt)
    estimator.update(measurements[t])
    out.append(estimator.position.clone())
  return torch.stack(out)


def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
  return float(torch.sqrt(torch.mean((a - b) ** 2)))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_shapes_and_finiteness():
  """Every estimator returns correctly shaped, finite tensors."""
  cfg = StateEstimatorCfg()
  B = 4
  for est in (
    ParticleFilter(B, cfg, num_particles=256, seed=0),
    ExtendedKalmanFilter(B, cfg),
    UnscentedKalmanFilter(B, cfg),
  ):
    assert est.state.shape == (B, 6), f"{est.name} state shape"
    assert est.covariance.shape == (B, 6, 6), f"{est.name} cov shape"
    assert est.position.shape == (B, 3), f"{est.name} position shape"
    assert est.position_std.shape == (B, 3), f"{est.name} std shape"
    est.step(0.02, torch.zeros(B, 3))
    assert torch.isfinite(est.state).all(), f"{est.name} produced non-finite state"
    assert torch.isfinite(est.covariance).all(), f"{est.name} non-finite cov"
  print("PASS  shapes and finiteness")


def test_filters_reduce_noise():
  """Each filter's output must be closer to truth than the raw measurement."""
  dt = 0.02
  truth, _ = make_trajectory(dt=dt)
  sigma = 0.01
  meas = add_noise(truth, sigma, seed=1)
  raw = rmse(meas, truth)

  cfg = StateEstimatorCfg(measurement_noise=sigma)
  B = truth.shape[1]
  results = {}
  for est in (
    ParticleFilter(B, cfg, num_particles=512, proposal="optimal", seed=0),
    ExtendedKalmanFilter(B, cfg),
    UnscentedKalmanFilter(B, cfg),
  ):
    est_rmse = rmse(run_filter(est, meas, dt), truth)
    results[est.name] = est_rmse
    assert est_rmse < raw, (
      f"{est.name} RMSE {est_rmse:.5f} is not better than raw {raw:.5f}"
    )
  print(f"PASS  noise reduction (raw RMSE {raw:.5f} m)")
  for name, val in results.items():
    print(f"        {name:<26} {val:.5f} m  ({100 * (1 - val / raw):.1f}% better)")


def test_ekf_ukf_agree():
  """On a linear-Gaussian system the EKF and UKF must agree closely.

  This is the correctness cross-check: two independent implementations of an
  optimal filter for the same linear system have to converge to the same
  answer. If they diverge, one of them has a bug.
  """
  dt = 0.02
  truth, _ = make_trajectory(steps=200)
  meas = add_noise(truth, 0.01, seed=2)
  cfg = StateEstimatorCfg(measurement_noise=0.01)
  B = truth.shape[1]

  ekf = run_filter(ExtendedKalmanFilter(B, cfg), meas, dt)
  ukf = run_filter(UnscentedKalmanFilter(B, cfg), meas, dt)

  # Skip the first few steps where sigma-point scaling differs transiently.
  diff = float(torch.max(torch.abs(ekf[10:] - ukf[10:])))
  assert diff < 1e-4, f"EKF and UKF disagree by {diff:.2e} m"
  print(f"PASS  EKF/UKF agreement (max divergence {diff:.2e} m)")


def test_particle_filter_beats_kalman_on_outliers():
  """The PF's advantage should appear under non-Gaussian (outlier) noise.

  This is the experiment that justifies the PF's extra compute for the
  report's research question. Under clean Gaussian noise the Kalman filters
  are optimal and the PF cannot beat them; under heavy-tailed noise the PF's
  ability to down-weight implausible measurements should win.

  Averaged over several independent noise realisations. A single realisation
  is not evidence: an earlier version of this test compared one seed and did
  not assert anything, which let a seed-dependent result look like a
  reproducible finding. If this claim goes in the report it has to hold on
  average, not on a lucky draw.
  """
  dt = 0.02
  truth, _ = make_trajectory(steps=600)
  sigma = 0.01
  B = truth.shape[1]

  # Deliberately tell the filters the *nominal* noise, not the outlier scale
  # -- this mirrors reality, where you tune for the sensor spec and the world
  # then hands you occasional garbage.
  cfg = StateEstimatorCfg(measurement_noise=sigma)

  pf_scores: list[float] = []
  ekf_scores: list[float] = []
  for seed in range(5):
    meas = add_noise(truth, sigma, outlier_rate=0.06, outlier_scale=30.0, seed=seed)
    pf = ParticleFilter(B, cfg, num_particles=512, proposal="optimal", seed=seed)
    pf_scores.append(rmse(run_filter(pf, meas, dt), truth))
    ekf_scores.append(rmse(run_filter(ExtendedKalmanFilter(B, cfg), meas, dt), truth))

  pf_mean = sum(pf_scores) / len(pf_scores)
  ekf_mean = sum(ekf_scores) / len(ekf_scores)
  pf_worst = max(pf_scores)

  print(
    f"PASS  outlier robustness  PF {pf_mean:.5f} m vs EKF {ekf_mean:.5f} m "
    f"(over {len(pf_scores)} seeds, PF worst {pf_worst:.5f} m)"
  )

  assert pf_mean < ekf_mean, (
    f"Particle filter should beat the EKF on average under outlier noise, "
    f"got PF {pf_mean:.5f} m vs EKF {ekf_mean:.5f} m. This is the result "
    f"the report's choice of a particle filter rests on -- if it fails, the "
    f"claim needs revisiting, not the threshold."
  )
  assert pf_worst < ekf_mean, (
    f"Particle filter lost on at least one seed (worst {pf_worst:.5f} m vs "
    f"EKF mean {ekf_mean:.5f} m); the advantage is not reproducible enough "
    f"to report as a general finding."
  )


def test_resampling_prevents_degeneracy():
  """ESS must stay healthy and the ensemble must keep distinct particles."""
  dt = 0.02
  truth, _ = make_trajectory(steps=300)
  meas = add_noise(truth, 0.01, seed=4)
  B = truth.shape[1]
  cfg = StateEstimatorCfg(measurement_noise=0.01)
  pf = ParticleFilter(B, cfg, num_particles=512, proposal="optimal", seed=0)
  pf.reset(measurement=meas[0])

  min_ess = float("inf")
  for t in range(meas.shape[0]):
    pf.step(dt, meas[t])
    min_ess = min(min_ess, float(pf.effective_sample_size.min()))

  unique = len(torch.unique(pf.particles[0, :, 0]))
  assert min_ess > 1.0, f"Particle set went fully degenerate (ESS {min_ess:.2f})"
  assert unique > 50, f"Only {unique} distinct particles left -- impoverishment"
  print(
    f"PASS  degeneracy control (min ESS {min_ess:.1f}/512, "
    f"{unique} distinct particles, {int(pf.resample_count[0])} resamples)"
  )


def test_partial_reset():
  """Resetting a subset of envs must not disturb the others."""
  cfg = StateEstimatorCfg()
  B = 6
  for est in (
    ParticleFilter(B, cfg, num_particles=128, seed=0),
    ExtendedKalmanFilter(B, cfg),
    UnscentedKalmanFilter(B, cfg),
  ):
    meas = torch.randn(B, 3) * 0.1 + torch.tensor([0.63, 0.0, 1.1])
    est.reset(measurement=meas)
    for _ in range(20):
      est.step(0.02, meas)
    before = est.state.clone()

    ids = torch.tensor([1, 4])
    est.reset(env_ids=ids, measurement=torch.zeros(2, 3))
    after = est.state

    untouched = torch.tensor([0, 2, 3, 5])
    assert torch.allclose(before[untouched], after[untouched], atol=1e-5), (
      f"{est.name} partial reset disturbed other envs"
    )
    assert float(torch.norm(after[ids])) < 0.5, f"{est.name} reset envs not cleared"
  print("PASS  partial reset isolation")


def test_handles_extreme_measurement():
  """A wildly wrong measurement must not produce NaN in any filter.

  This is the underflow case the log-space weighting exists to survive: a
  measurement 50 m away makes every particle's linear likelihood exactly 0.
  """
  cfg = StateEstimatorCfg(measurement_noise=0.005)
  B = 2
  for est in (
    ParticleFilter(B, cfg, num_particles=256, seed=0),
    ExtendedKalmanFilter(B, cfg),
    UnscentedKalmanFilter(B, cfg),
  ):
    est.reset(measurement=torch.tensor([[0.63, 0.0, 1.1]] * B))
    est.step(0.02, torch.full((B, 3), 50.0))
    assert torch.isfinite(est.state).all(), f"{est.name} produced NaN/Inf"
    est.step(0.02, torch.tensor([[0.63, 0.0, 1.1]] * B))
    assert torch.isfinite(est.state).all(), f"{est.name} did not recover"
  print("PASS  extreme measurement robustness")


def test_optimal_proposal_beats_bootstrap():
  """Regression guard for the velocity-observability failure.

  With a physically-correct (narrow) process noise, the bootstrap proposal
  cannot track velocity from position-only measurements and lags badly. The
  optimal proposal must fix this at the same particle count. If this test
  ever fails, check that ParticleFilter.proposal still defaults to "optimal".
  """
  dt = 0.02
  truth, _ = make_trajectory(dt=dt)
  sigma = 0.01
  meas = add_noise(truth, sigma, seed=1)
  raw = rmse(meas, truth)
  cfg = StateEstimatorCfg(measurement_noise=sigma)  # physically-motivated Q
  B = truth.shape[1]

  boot = rmse(
    run_filter(
      ParticleFilter(B, cfg, num_particles=512, proposal="bootstrap", seed=0), meas, dt
    ),
    truth,
  )
  opt = rmse(
    run_filter(
      ParticleFilter(B, cfg, num_particles=512, proposal="optimal", seed=0), meas, dt
    ),
    truth,
  )
  assert opt < boot, f"optimal ({opt:.5f}) should beat bootstrap ({boot:.5f})"
  assert opt < raw, f"optimal PF ({opt:.5f}) must beat raw measurement ({raw:.5f})"
  print(
    f"PASS  proposal comparison  optimal {opt:.5f} m vs "
    f"bootstrap {boot:.5f} m (raw {raw:.5f} m)"
  )


def test_ukf_rejects_unstable_alpha():
  """The UKF must refuse an alpha that is unusable in float32.

  alpha=1e-3 is the textbook value but silently destroys accuracy at float32
  precision, so it is rejected loudly rather than producing quiet garbage.
  """
  cfg = StateEstimatorCfg()
  try:
    UnscentedKalmanFilter(2, cfg, alpha=1e-3)
  except ValueError as exc:
    assert "float32" in str(exc)
    print("PASS  UKF rejects numerically unstable alpha")
    return
  raise AssertionError("UKF accepted alpha=1e-3 without complaint")


if __name__ == "__main__":
  torch.manual_seed(0)
  test_shapes_and_finiteness()
  test_filters_reduce_noise()
  test_ekf_ukf_agree()
  test_optimal_proposal_beats_bootstrap()
  test_particle_filter_beats_kalman_on_outliers()
  test_resampling_prevents_degeneracy()
  test_partial_reset()
  test_handles_extreme_measurement()
  test_ukf_rejects_unstable_alpha()
  print("\nAll state-estimation tests passed.")
