"""Single source of truth for the whiteboard geometry.

These values were previously duplicated across four files -- DrawingCanvas.py,
draw_target_cmd.py, drawing_env_cfg.py and config/g1/env_cfgs.py -- each with
its own slightly different Y and Z ranges (+/-0.40, +/-0.35 and +/-0.30 all
appeared). That is a silent-failure hazard: move the board in the MJCF, miss one
file, and training carries on happily while some fraction of draw targets sit
behind or beside the board where the arm can never reach them. The reward just
quietly stops being achievable and the policy plateaus for no visible reason.

Everything now imports from here. If the board moves in the MJCF, change these
constants and nothing else.

MJCF reference (g1_29dof_whiteboard.xml)::

    body pos   = (0.35, 0, 1.10)
    geom size  = (0.02, 0.40, 0.35)      half-extents in (x, y, z)

so the writable face is at x = 0.35 - 0.02 = 0.33, and the physical surface
spans y in [-0.40, +0.40], z in [0.75, 1.45].
"""

from __future__ import annotations

# --- Physical board surface (must match the MJCF) --------------------------

BOARD_CENTRE_X: float = 0.35
BOARD_CENTRE_Y: float = 0.0
BOARD_CENTRE_Z: float = 1.10

BOARD_HALF_DEPTH: float = 0.02
BOARD_HALF_WIDTH: float = 0.40
BOARD_HALF_HEIGHT: float = 0.35

#: World/env-local X of the face the robot writes on (the -X face).
BOARD_FACE_X: float = BOARD_CENTRE_X - BOARD_HALF_DEPTH  # 0.33
WRITING_X: float = BOARD_FACE_X - 0.008

#: Full physical extent of the drawable surface.
BOARD_Y_MIN: float = BOARD_CENTRE_Y - BOARD_HALF_WIDTH  # -0.40
BOARD_Y_MAX: float = BOARD_CENTRE_Y + BOARD_HALF_WIDTH  # +0.40
BOARD_Z_MIN: float = BOARD_CENTRE_Z - BOARD_HALF_HEIGHT  #  0.75
BOARD_Z_MAX: float = BOARD_CENTRE_Z + BOARD_HALF_HEIGHT  #  1.45


# --- Target sampling region (a margin inside the physical surface) ---------
#
# Targets are deliberately sampled inside the physical edge. A target exactly on
# the boundary is reachable only with the pen perfectly perpendicular, so edge
# targets produce a stream of near-unreachable goals that add reward variance
# without teaching anything useful. The margin is the tuning knob: widen it as
# the policy improves.

TARGET_MARGIN_Y: float = 0.10
TARGET_MARGIN_Z: float = 0.05

TARGET_Y_RANGE: tuple[float, float] = (
  BOARD_Y_MIN + TARGET_MARGIN_Y,  # -0.30
  BOARD_Y_MAX - TARGET_MARGIN_Y,  # +0.30
)
TARGET_Z_RANGE: tuple[float, float] = (
  BOARD_Z_MIN + TARGET_MARGIN_Z,  # 0.80
  BOARD_Z_MAX - TARGET_MARGIN_Z,  # 1.40
)


# --- Pen contact ------------------------------------------------------------

#: How close in X the pen tip must be to the face to count as touching.
CONTACT_X_THRESHOLD: float = 0.015


def validate() -> None:
  """Assert the derived ranges are self-consistent.

  Called at import time by the env config so a bad edit fails immediately at
  build time rather than 200k training steps later.
  """
  assert BOARD_Y_MIN < BOARD_Y_MAX, "Board Y extent is inverted."
  assert BOARD_Z_MIN < BOARD_Z_MAX, "Board Z extent is inverted."
  assert TARGET_Y_RANGE[0] < TARGET_Y_RANGE[1], (
    f"TARGET_MARGIN_Y={TARGET_MARGIN_Y} is too large for the board width; "
    f"the sampling range collapsed to {TARGET_Y_RANGE}."
  )
  assert TARGET_Z_RANGE[0] < TARGET_Z_RANGE[1], (
    f"TARGET_MARGIN_Z={TARGET_MARGIN_Z} is too large for the board height; "
    f"the sampling range collapsed to {TARGET_Z_RANGE}."
  )
  assert BOARD_Y_MIN <= TARGET_Y_RANGE[0] and TARGET_Y_RANGE[1] <= BOARD_Y_MAX, (
    "Target Y sampling range extends beyond the physical board."
  )
  assert BOARD_Z_MIN <= TARGET_Z_RANGE[0] and TARGET_Z_RANGE[1] <= BOARD_Z_MAX, (
    "Target Z sampling range extends beyond the physical board."
  )


validate()
