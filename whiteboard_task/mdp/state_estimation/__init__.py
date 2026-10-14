"""Probabilistic state estimation for the G1 whiteboard drawing task.

Implements the framework described in Section 4 of the interim report: a
particle filter as the primary estimator, plus EKF and UKF baselines for the
comparative benchmarking of Section 4.3.

Quick start -- run the filter comparison without needing mjlab::

    python -m mdp.state_estimation.benchmark

Attach an estimator to the drawing environment by adding the observation terms
from ``mdp.observations`` (``estimated_pen_position`` and friends) to the
observation group; the filter is created and stepped automatically.
"""

from .base import (
  MEAS_DIM,
  POS_SLICE,
  STATE_DIM,
  VEL_SLICE,
  StateEstimator,
  StateEstimatorCfg,
  constant_velocity_transition,
  position_measurement_matrix,
)
from .benchmark import (
  BenchmarkResult,
  corrupt,
  evaluate,
  format_table,
  generate_trajectory,
  run_benchmark,
)
from .kalman import ExtendedKalmanFilter, UnscentedKalmanFilter
from .particle_filter import ParticleFilter

# The integration layer needs mjlab; the filters and the benchmark do not.
# Guarding the import keeps `python -m mdp.state_estimation.benchmark` working
# on a machine without mjlab installed -- which is the whole point of having a
# standalone benchmark. If mjlab is missing, the env-facing names simply are
# not exported and anything that needs them fails with a clear ImportError at
# the point of use rather than at package import.
try:
  from .integration import (
    FilterType,
    StateEstimationCfg,
    StateEstimatorManager,
    build_estimator,
    get_state_estimator,
    has_state_estimator,
    reset_state_estimator,
  )

  _MJLAB_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on install environment
  _MJLAB_AVAILABLE = False

__all__ = [
  # Core interface
  "StateEstimator",
  "StateEstimatorCfg",
  "STATE_DIM",
  "MEAS_DIM",
  "POS_SLICE",
  "VEL_SLICE",
  "constant_velocity_transition",
  "position_measurement_matrix",
  # Estimators
  "ParticleFilter",
  "ExtendedKalmanFilter",
  "UnscentedKalmanFilter",
  # Benchmarking
  "BenchmarkResult",
  "run_benchmark",
  "evaluate",
  "generate_trajectory",
  "corrupt",
  "format_table",
]

if _MJLAB_AVAILABLE:
  __all__ += [
    # Environment integration (requires mjlab)
    "StateEstimationCfg",
    "StateEstimatorManager",
    "FilterType",
    "build_estimator",
    "get_state_estimator",
    "has_state_estimator",
    "reset_state_estimator",
  ]
