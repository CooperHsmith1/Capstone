"""Unitree G1 whiteboard drawing environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.whiteboard.drawing_env_cfg import make_drawing_env_cfg
from mjlab.tasks.whiteboard.mdp.board import (
  BOARD_FACE_X,
  TARGET_Y_RANGE,
  TARGET_Z_RANGE,
)
from mjlab.tasks.whiteboard.mdp.draw_target_cmd import DrawTargetCommandCfg
from mjlab.tasks.whiteboard.mdp.state_estimation import StateEstimationCfg


def unitree_g1_drawing_env_cfg(
  play: bool = False,
  num_envs: int = 4096,
  fixed_base: bool = True,
  state_estimation: StateEstimationCfg | None = None,
) -> ManagerBasedRlEnvCfg:
  """Create the Unitree G1 whiteboard drawing environment configuration.

  Training stages (controlled by the command resampling range):
    Early  : targets held 4-8 s so the arm has time to converge.
    Later  : tighten to 2-4 s once the policy reliably reaches points.
    Future : swap DrawTargetCommandCfg for a path-following command to draw
             shapes and letters.

  Args:
    play: Play/eval mode -- disables observation corruption, extends the
      episode, and holds each target longer so the arm can be watched.
    num_envs: Parallel environments. Override from the CLI with
      ``--env.scene.num-envs``.
    fixed_base: Bolt the robot to the world. Recommended until the reaching
      behaviour is learned.
    state_estimation: Route the pen observation through a particle filter,
      EKF or UKF instead of ground truth. See Section 4 of the interim report.

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
  cfg = make_drawing_env_cfg(
    num_envs=num_envs,
    fixed_base=fixed_base,
    state_estimation=state_estimation,
  )

  if play:
    cfg.scene.num_envs = min(num_envs, 16)
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    # Hold each target longer in play so the arm can be watched reaching it.
    cfg.commands["draw_target"] = DrawTargetCommandCfg(
      resampling_time_range=(6.0, 10.0),
      board_face_x=BOARD_FACE_X,
      board_y_range=TARGET_Y_RANGE,
      board_z_range=TARGET_Z_RANGE,
    )

  return cfg
