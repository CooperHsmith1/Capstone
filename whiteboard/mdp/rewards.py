from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .cfg_utils import resolve_first_site_id

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pen_tip_pos_local(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Return ENV-LOCAL pen_tip site position, shape [B, 3].

  site_pos_w is the ABSOLUTE world position, which for env N includes that
  env's tiled spawn offset (env_origins) — tens of metres for far envs. The
  draw target is specified in env-local coords (0.322, y, z), so we MUST
  subtract env_origins to compare them in the same frame. Without this the
  pen-to-target distance reads ~47 m and pen_tracking is always ~0.
  """
  asset: Entity = env.scene[asset_cfg.name]
  idx = resolve_first_site_id(asset_cfg)  # int — never a bare slice
  pos_w = asset.data.site_pos_w[:, idx, :]  # [B, 3] absolute world
  return pos_w - env.scene.env_origins  # [B, 3] env-local


# ---------------------------------------------------------------------------
# DRAWING REWARDS
# ---------------------------------------------------------------------------


def pen_tracking_reward(
  env: "ManagerBasedRlEnv",
  command_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reward the pen tip for being close to the current draw-target command.

  Args:
      command_name: Key in the command manager, e.g. ``"draw_target"``.
      asset_cfg:    SceneEntityCfg for "robot" with site_names=("pen_tip",).
  """
  pen_pos = _pen_tip_pos_local(env, asset_cfg)  # [B, 3] env-local
  target = env.command_manager.get_command(command_name)  # [B, 3] env-local
  assert target is not None, (
    f"Command '{command_name}' returned None. "
    "Check that DrawTargetCommandCfg is registered in the env commands dict."
  )
  dist = torch.norm(pen_pos - target, dim=1)  # [B]
  # A sharper falloff gives the policy useful gradient when it is still
  # 0.3-0.5 m from the board instead of making that whole region look similar.
  return torch.exp(-6.0 * dist)


def pen_approach_reward(
  env: "ManagerBasedRlEnv",
  writing_x: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reward moving the pen toward the board independently of Y/Z tracking."""
  pen_pos = _pen_tip_pos_local(env, asset_cfg)
  return torch.exp(-4.0 * torch.abs(pen_pos[:, 0] - writing_x))


def pen_penetration_penalty(
  env: "ManagerBasedRlEnv",
  whiteboard_x: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Return pen depth past the physical board face for a negative reward."""
  pen_pos = _pen_tip_pos_local(env, asset_cfg)
  return torch.relu(pen_pos[:, 0] - whiteboard_x)


def pen_contact_reward(
  env: "ManagerBasedRlEnv",
  whiteboard_x: float,
  asset_cfg: SceneEntityCfg,
  writing_x: float | None = None,
) -> torch.Tensor:
  """Reward contact at the writing plane without board penetration.

  Args:
      whiteboard_x: Env-local X coordinate of the physical board face.
      asset_cfg:    SceneEntityCfg with site_names=("pen_tip",).
      writing_x: Env-local X coordinate of the writing plane. If omitted, the
        physical face is used for backwards compatibility.
  """
  pen_pos = _pen_tip_pos_local(env, asset_cfg)  # [B, 3] env-local
  target_x = whiteboard_x if writing_x is None else writing_x
  x_error = torch.abs(pen_pos[:, 0] - target_x)
  penetration = torch.relu(pen_pos[:, 0] - whiteboard_x)
  return torch.exp(-40.0 * x_error) * torch.exp(-80.0 * penetration)


def smooth_pen_motion_reward(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalise high joint velocities in the drawing arm.

  Returns mean-squared joint velocity (positive). Use a negative weight
  in RewardTermCfg to turn this into a penalty.

  Args:
      asset_cfg: SceneEntityCfg with joint_names resolving the right-arm joints.
  """
  asset: Entity = env.scene[asset_cfg.name]
  # joint_ids is list[int] | slice — torch accepts both for tensor indexing.
  joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]  # [B, J]
  return torch.mean(torch.square(joint_vel), dim=1)  # [B]


# ---------------------------------------------------------------------------
# STABILITY REWARDS
# ---------------------------------------------------------------------------


def upright_reward(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reward the robot for staying upright (projected gravity near [0, 0, -1])."""
  asset: Entity = env.scene[asset_cfg.name]
  up = asset.data.projected_gravity_b  # [B, 3]
  error = torch.sum(torch.square(up[:, :2]), dim=1)  # [B]
  return torch.exp(-5.0 * error)


# ---------------------------------------------------------------------------
# LOCOMOTION REWARDS (kept for future use)
# ---------------------------------------------------------------------------


def track_linear_velocity(
  env: "ManagerBasedRlEnv",
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' returned None."
  actual = asset.data.root_link_lin_vel_b
  error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  return torch.exp(-error / std**2)


def track_angular_velocity(
  env: "ManagerBasedRlEnv",
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' returned None."
  actual = asset.data.root_link_ang_vel_b
  error = torch.square(command[:, 2] - actual[:, 2]) + torch.sum(
    torch.square(actual[:, :2]), dim=1
  )
  return torch.exp(-error / std**2)
