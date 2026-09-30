"""Continuous Gerono figure-eight target command for whiteboard drawing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .board import (
  BOARD_CENTRE_Y,
  BOARD_CENTRE_Z,
  BOARD_FACE_X,
  TARGET_Y_RANGE,
  TARGET_Z_RANGE,
  WRITING_X,
)
from .cfg_utils import resolve_first_site_id

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def gerono_figure_eight(
  phase: torch.Tensor,
  center_y: float,
  center_z: float,
  radius: float,
) -> torch.Tensor:
  """Return the figure-eight Y/Z coordinates for each phase."""
  return torch.stack(
    (
      center_y + radius * torch.sin(phase),
      center_z + 0.5 * radius * torch.sin(2.0 * phase),
    ),
    dim=-1,
  )


class DrawTargetCommand(CommandTerm):
  """Moves a centered pen target through one continuous Gerono figure eight."""

  cfg: DrawTargetCommandCfg

  def __init__(self, cfg: DrawTargetCommandCfg, env: ManagerBasedRlEnv):
    # NOTE ON ORDERING: CommandTerm.__init__ may call _update_metrics() or
    # _resample_command() before returning. Both touch self._target, so it
    # must exist *before* super().__init__ runs, not after. Creating it
    # afterwards worked only by luck of the base-class implementation and
    # would break on an mjlab upgrade with an AttributeError that points at
    # this file but not at the real cause.
    num_envs = int(env.num_envs)
    device = env.device

    # Resolved lazily on first use and cached — see _get_pen_tip_pos.
    self._pen_site_idx: int | None = None

    # [B, 3] — fixed writing-plane X and the animated Y/Z target.
    self._target = torch.zeros(num_envs, 3, device=device)
    self._phase = torch.zeros(num_envs, device=device)
    self._elapsed = torch.zeros(num_envs, device=device)
    self._target[:, 0] = cfg.writing_x

    super().__init__(cfg, env)

    # Per-env metric tensor MUST exist (shape [B]) before any reset, and must
    # never be a scalar. The command manager indexes it as metric[env_ids],
    # so a 0-dim (scalar) value raises "too many indices for tensor of
    # dimension 0". Initialise it here as a proper per-environment tensor.
    self.metrics["pen_dist_to_target"] = torch.zeros(self.num_envs, device=self.device)

    # Start at the center target so the arm can reach before drawing begins.
    self._resample_command(torch.arange(self.num_envs, device=self.device))

  # ------------------------------------------------------------------
  # CommandTerm interface
  # ------------------------------------------------------------------

  @property
  def command(self) -> torch.Tensor:
    """Return target positions, shape [B, 3]."""
    return self._target

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._elapsed[env_ids] = 0.0
    self._phase[env_ids] = 0.0
    self._set_figure_eight_target(env_ids)

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self._elapsed[ids] += self._env.step_dt
    self._phase[ids] = (self._elapsed[ids] - self.cfg.approach_duration_s).clamp_min(
      0.0
    ) * self.cfg.angular_speed
    self._set_figure_eight_target(env_ids)

  def _set_figure_eight_target(self, env_ids: torch.Tensor | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self._target[ids, 0] = self.cfg.writing_x
    self._target[ids, 1:] = gerono_figure_eight(
      self._phase[ids],
      self.cfg.center_y,
      self.cfg.center_z,
      self.cfg.radius,
    )

  def _update_metrics(self) -> None:
    pen_pos = self._get_pen_tip_pos()
    if pen_pos is not None:
      # Keep this PER-ENV (shape [B]); do NOT call .mean() — the command
      # manager indexes the metric as metric[env_ids] on reset.
      self.metrics["pen_dist_to_target"] = torch.norm(pen_pos - self._target, dim=1)
    # If pen_pos is None (early setup), keep the existing per-env tensor
    # already created in __init__ (shape [B]).

  # ------------------------------------------------------------------
  # Helpers
  # ------------------------------------------------------------------

  def _get_pen_tip_pos(self) -> torch.Tensor | None:
    """Try to read pen_tip site position from the robot.

    Returns ``None`` on failure so ``_update_metrics`` degrades gracefully
    rather than crashing during early env setup, before the scene has
    finished building.

    The site index is resolved once and cached. The previous version built a
    fresh SceneEntityCfg and re-resolved the site name on *every* call, i.e.
    once per environment per step -- a string lookup and an allocation in
    the innermost training loop, for a value that never changes.
    """
    if self._pen_site_idx is None:
      try:
        asset_cfg = SceneEntityCfg("robot", site_names=("pen_tip",))
        asset_cfg.resolve(self._env.scene)
        # resolve_first_site_id handles list[int] | slice.
        self._pen_site_idx = resolve_first_site_id(asset_cfg)
      except (KeyError, AttributeError, ValueError):
        # Scene not ready yet; try again on a later step.
        return None

    try:
      asset = self._env.scene["robot"]
      pos_w = asset.data.site_pos_w[:, self._pen_site_idx, :]  # world
      # Subtract per-env tiled offset so this matches the env-local target.
      return pos_w - self._env.scene.env_origins  # env-local
    except (KeyError, AttributeError, IndexError):
      return None


@dataclass(kw_only=True)
class DrawTargetCommandCfg(CommandTermCfg):
  """Configuration for :class:`DrawTargetCommand`."""

  board_face_x: float = BOARD_FACE_X
  writing_x: float = WRITING_X
  board_y_range: tuple[float, float] = TARGET_Y_RANGE
  board_z_range: tuple[float, float] = TARGET_Z_RANGE
  center_y: float = BOARD_CENTRE_Y
  center_z: float = BOARD_CENTRE_Z
  approach_duration_s: float = 3.0
  radius: float = 0.15
  angular_speed: float = 0.35

  def build(self, env: ManagerBasedRlEnv) -> DrawTargetCommand:
    return DrawTargetCommand(self, env)
