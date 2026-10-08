from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor.contact_sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

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
  draw target is specified in env-local coords (0.63, y, z), so we MUST
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
  sharpness: float = 2.0,
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
  return torch.exp(-sharpness * dist)


def pen_distance_penalty(
  env: "ManagerBasedRlEnv",
  command_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Pen-to-target distance in metres, [B]. Use a negative weight.

  Unlike the exponential tracking terms this has the same gradient at any
  range, so a pen folded against the chest still feels a pull toward the board.
  """
  pen_pos = _pen_tip_pos_local(env, asset_cfg)
  target = env.command_manager.get_command(command_name)
  return torch.norm(pen_pos - target, dim=1)


def walk_to_board_reward(
  env: "ManagerBasedRlEnv",
  board_x: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reward the floating robot for moving its torso toward the board."""
  asset: Entity = env.scene[asset_cfg.name]
  torso_x = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  return torch.exp(-8.0 * torch.abs(torso_x - board_x))


def pen_contact_reward(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  command_name: str | None = None,
  max_force: float | None = None,
) -> torch.Tensor:
  """Reward actual pen-tip contact with the board, shape [B].

  With ``max_force`` set, only light contact counts, so pressing hard (leaning
  on the pen) earns nothing.

  With ``command_name`` set, contact is rewarded while the draw command wants
  the pen down and penalised while it wants the pen lifted (G-code travel).
  """
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  touching = found.any(dim=1).to(dtype=torch.float32)
  if max_force is not None:
    force = sensor.data.force
    assert force is not None
    touching = touching * (force[..., 0].sum(dim=1).abs() <= max_force).float()
  if command_name is None:
    return touching
  command = env.command_manager.get_term(command_name)
  sign = 2.0 * command.pen_down.to(dtype=torch.float32) - 1.0  # type: ignore[attr-defined]
  return touching * sign


def pen_pressing_too_hard(
  env: "ManagerBasedRlEnv", sensor_name: str, max_force: float
) -> torch.Tensor:
  """True where the pen pushes on the board harder than ``max_force``, [B] bool.

  Used as a termination so the pen cannot be used as a prop to lean on.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  return force[..., 0].sum(dim=1).abs() > max_force


def body_touching_board(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
  """True where the torso or pelvis touches the board, shape [B] bool.

  Used as both a penalty and a termination so the policy cannot stay upright
  by leaning on the board.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  return found.any(dim=1)


def base_height_reward(
  env: "ManagerBasedRlEnv",
  target_height: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reward keeping the pelvis near standing height (blocks sitting/splits)."""
  asset: Entity = env.scene[asset_cfg.name]
  height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  return torch.exp(-10.0 * torch.abs(height - target_height))


def path_progress_reward(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  """Dense reward for how far along the G-code path the pen has got, [B]."""
  command = env.command_manager.get_term(command_name)
  return command.progress  # type: ignore[attr-defined]


def pen_contact_force_penalty(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  max_normal_force: float = 5.0,
) -> torch.Tensor:
  """Penalize excessive board-normal force only while the tip is in contact."""
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  force = sensor.data.force
  assert found is not None
  assert force is not None
  in_contact = found.any(dim=1).to(dtype=force.dtype)
  normal_force = force[..., 0].sum(dim=1).abs()
  excess = ((normal_force - max_normal_force) / max_normal_force).clamp(0.0, 1.0)
  return in_contact * excess.square()


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
  sharpness: float = 5.0,
) -> torch.Tensor:
  """Reward the robot for staying upright (projected gravity near [0, 0, -1])."""
  asset: Entity = env.scene[asset_cfg.name]
  up = asset.data.projected_gravity_b  # [B, 3]
  error = torch.sum(torch.square(up[:, :2]), dim=1)  # [B]
  return torch.exp(-sharpness * error)


def pelvis_upright_reward(
  env: "ManagerBasedRlEnv", sharpness: float = 20.0
) -> torch.Tensor:
  """Reward a level pelvis, which stops the hips-back / shoulders-back lean."""
  asset: Entity = env.scene["robot"]
  up = asset.data.projected_gravity_b  # root (pelvis) frame
  return torch.exp(-sharpness * torch.sum(torch.square(up[:, :2]), dim=1))


def arms_behind_body_penalty(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg,
  min_forward: float = 0.0,
) -> torch.Tensor:
  """Penalise arm links that sit behind the pelvis, shape [B].

  Positions are taken in the pelvis frame (+X forward). Each link contributes
  the squared distance it is behind ``min_forward``.
  """
  asset: Entity = env.scene[asset_cfg.name]
  rel = asset.data.body_link_pos_w[:, asset_cfg.body_ids] - (
    asset.data.root_link_pos_w.unsqueeze(1)
  )
  quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, rel.shape[1], -1)
  forward = quat_apply_inverse(quat, rel)[..., 0]
  return torch.sum((min_forward - forward).clamp_min(0.0).square(), dim=1)


def arm_trapped(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg,
  max_behind: float = 0.15,
  min_lateral: float = 0.06,
) -> torch.Tensor:
  """True where an arm link is far behind the pelvis or jammed against the torso.

  A link counts as jammed when it is behind the pelvis and within
  ``min_lateral`` of the body midline (arm wedged between hip and torso).
  """
  asset: Entity = env.scene[asset_cfg.name]
  rel = asset.data.body_link_pos_w[:, asset_cfg.body_ids] - (
    asset.data.root_link_pos_w.unsqueeze(1)
  )
  quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, rel.shape[1], -1)
  local = quat_apply_inverse(quat, rel)
  behind = local[..., 0] < -max_behind
  jammed = (local[..., 0] < -0.03) & (local[..., 1].abs() < min_lateral)
  return (behind | jammed).any(dim=1)


def waist_deviation_penalty(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Squared deviation of the waist joints from the reset pose, shape [B]."""
  asset: Entity = env.scene[asset_cfg.name]
  q = asset.data.joint_pos[:, asset_cfg.joint_ids]
  q0 = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  return torch.sum(torch.square(q - q0), dim=1)


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
