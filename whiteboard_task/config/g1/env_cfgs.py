"""Unitree G1 whiteboard drawing environment configurations."""

import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.whiteboard.drawing_env_cfg import make_drawing_env_cfg
from mjlab.tasks.whiteboard.mdp.state_estimation import StateEstimationCfg


def unitree_g1_drawing_env_cfg(
  play: bool = False,
  num_envs: int = 4096,
  fixed_base: bool = False,
  state_estimation: StateEstimationCfg | None = None,
  gcode_path: str | None = None,
) -> ManagerBasedRlEnvCfg:
  """Create the Unitree G1 whiteboard drawing environment configuration.

  The target holds at the board center while the robot approaches, then follows
  a complete Gerono figure-eight. Training and playback use the same path.

  Args:
    play: Play/eval mode -- disables observation corruption, extends the
      episode, and holds each target longer so the arm can be watched.
    num_envs: Parallel environments. Override from the CLI with
      ``--env.scene.num-envs``.
    fixed_base: Bolt the robot to the world. Recommended until the reaching
      behaviour is learned.
    state_estimation: Route the pen observation through a particle filter,
      EKF or UKF instead of ground truth. See Section 4 of the interim report.
    gcode_path: G-code file to draw instead of the figure eight. Defaults to the
      ``MJLAB_DRAWING_GCODE`` environment variable, which is how the ``train``
      and ``play`` scripts pick it up.

  Note:
    Reward terms are deliberately NOT rebuilt here. An earlier version of this
    function reassigned pen_tracking, pen_contact, smooth_pen_motion and
    upright from scratch, which silently overwrote the tuned weights in
    drawing_env_cfg.py with older pre-tuning values (smooth_pen_motion -0.01 ->
    -0.05, upright 1.0 -> 2.0). Because this function is what actually gets
    registered, the tuning never took effect in any training run. Override
    individual ``.weight`` fields if needed, but never reassign the whole
    RewardTermCfg.
  """
  gcode_path = gcode_path or os.environ.get("MJLAB_DRAWING_GCODE") or None
  cfg = make_drawing_env_cfg(
    num_envs=num_envs,
    fixed_base=fixed_base,
    state_estimation=state_estimation,
    gcode_path=gcode_path,
  )

  if play:
    cfg.scene.num_envs = min(num_envs, 16)
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
  return cfg
