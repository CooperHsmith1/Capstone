from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class CircleDrawingCommand(CommandTerm):
  """World-frame point that advances around a circular drawing target."""

  cfg: CircleDrawingCommandCfg

  def __init__(self, cfg: CircleDrawingCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self.phase = torch.zeros(self.num_envs, device=self.device)
    self.episode_progress = torch.zeros(self.num_envs, device=self.device)
    self.completion_pulse = torch.zeros(self.num_envs, device=self.device)
    self.metrics["tracking_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["circle_progress"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos

  def _update_metrics(self) -> None:
    self.metrics["tracking_error"] = torch.zeros_like(self.phase)
    self.metrics["circle_progress"] = self.episode_progress

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.phase[env_ids] = 0.0
    self.episode_progress[env_ids] = 0.0
    self.completion_pulse[env_ids] = 0.0
    self._set_target(env_ids)

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is not None:
      self._set_target(env_ids)
      return
    self.phase.add_(self.cfg.angular_speed * self._env.step_dt)
    completed = self.phase >= 2.0 * math.pi
    self.completion_pulse = completed.float()
    self.episode_progress.add_(completed.float())
    self.phase.remainder_(2.0 * math.pi)
    self._set_target(torch.arange(self.num_envs, device=self.device))

  def _set_target(self, env_ids: torch.Tensor) -> None:
    angle = self.phase[env_ids]
    center = torch.tensor(self.cfg.center, device=self.device).expand(len(env_ids), 3)
    center = center + self._env.scene.env_origins[env_ids]
    target = center.clone()
    target[:, self.cfg.plane_axes[0]] += self.cfg.radius * torch.cos(angle)
    target[:, self.cfg.plane_axes[1]] += self.cfg.radius * torch.sin(angle)
    target[:, self.cfg.normal_axis] += self.cfg.surface_offset
    self.target_pos[env_ids] = target


@dataclass(kw_only=True)
class CircleDrawingCommandCfg(CommandTermCfg):
  center: tuple[float, float, float] = (0.32, -0.02, 0.35)
  radius: float = 0.12
  angular_speed: float = 1.5
  surface_offset: float = 0.0
  plane_axes: tuple[int, int] = (0, 2)
  normal_axis: int = 1

  def __post_init__(self) -> None:
    if self.radius <= 0.0:
      raise ValueError("Circle radius must be positive")
    if self.angular_speed <= 0.0:
      raise ValueError("Circle angular speed must be positive")
    if len(set(self.plane_axes)) != 2 or self.normal_axis in self.plane_axes:
      raise ValueError("Circle axes must be two distinct axes excluding normal_axis")

  def build(self, env: ManagerBasedRlEnv) -> CircleDrawingCommand:
    return CircleDrawingCommand(self, env)
