"""MDP terms for the G1 whiteboard drawing task.

Import structure note: most terms in this package need mjlab, but the state
estimators and their benchmark deliberately do not. The mjlab-dependent imports
below are therefore guarded, so that

    python -m mdp.state_estimation.benchmark

still runs on a machine without mjlab installed -- which is the point of having
a standalone benchmark you can run while the sim is training elsewhere. When
mjlab is absent, the env-facing names are simply not exported; anything that
needs them raises a clear ImportError at the point of use.
"""

# --- Always available (pure torch / stdlib) --------------------------------

from .board import (
  BOARD_FACE_X,
  BOARD_Y_MAX,
  BOARD_Y_MIN,
  BOARD_Z_MAX,
  BOARD_Z_MIN,
  CONTACT_X_THRESHOLD,
  TARGET_Y_RANGE,
  TARGET_Z_RANGE,
)
from .state_estimation import (
  ExtendedKalmanFilter,
  ParticleFilter,
  StateEstimatorCfg,
  UnscentedKalmanFilter,
)

__all__ = [
  # Board geometry
  "BOARD_FACE_X",
  "BOARD_Y_MIN",
  "BOARD_Y_MAX",
  "BOARD_Z_MIN",
  "BOARD_Z_MAX",
  "CONTACT_X_THRESHOLD",
  "TARGET_Y_RANGE",
  "TARGET_Z_RANGE",
  # Estimators (no mjlab needed)
  "ParticleFilter",
  "ExtendedKalmanFilter",
  "UnscentedKalmanFilter",
  "StateEstimatorCfg",
]

# --- Requires mjlab --------------------------------------------------------

try:
  from .cfg_utils import resolve_first_id, resolve_first_site_id
  from .draw_target_cmd import DrawTargetCommandCfg
  from .observations import (
    estimated_pen_position,
    estimated_pen_velocity,
    pen_estimate_innovation,
    pen_position_uncertainty,
    site_position,
  )
  from .rewards import (
    pen_approach_reward,
    pen_contact_reward,
    pen_penetration_penalty,
    pen_tracking_reward,
    smooth_pen_motion_reward,
    upright_reward,
  )
  from .state_estimation import (
    StateEstimationCfg,
    get_state_estimator,
    reset_state_estimator,
  )

  _MJLAB_AVAILABLE = True

  __all__ += [
    # Commands
    "DrawTargetCommandCfg",
    # Observations
    "site_position",
    "estimated_pen_position",
    "estimated_pen_velocity",
    "pen_position_uncertainty",
    "pen_estimate_innovation",
    # Rewards
    "pen_tracking_reward",
    "pen_approach_reward",
    "pen_contact_reward",
    "pen_penetration_penalty",
    "smooth_pen_motion_reward",
    "upright_reward",
    # State estimation env integration
    "StateEstimationCfg",
    "get_state_estimator",
    "reset_state_estimator",
    # Utils
    "resolve_first_id",
    "resolve_first_site_id",
  ]
except ImportError:  # pragma: no cover - depends on install environment
  _MJLAB_AVAILABLE = False
