"""DrawTargetCommand — moves a 3-D target around a circle on the whiteboard.

The whiteboard in the MJCF lives at world position (0.35, 0, 1.1) with a
half-extent of (0.02, 0.4, 0.35) in (x, y, z).  The face the robot writes on
is the -X face, so all targets share the constant X value of the writing plane
(``0.322``).  Y and Z follow a configurable circular path within the board
bounds.

Usage in env cfg::

    commands: dict[str, CommandTermCfg] = {
        "draw_target": DrawTargetCommandCfg(
            radius=0.15,
            angular_speed=0.8,
        ),
    }

The command tensor has shape [B, 3] and is exposed via
``env.command_manager.get_command("draw_target")``.
"""

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


class DrawTargetCommand(CommandTerm):
  """Moves the target continuously around a circular path on the board."""

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

    # [B, 3]  —  (x=board_face_x, y=uniform, z=uniform)
    self._target = torch.zeros(num_envs, 3, device=device)
    self._phase = torch.zeros(num_envs, device=device)
    self._target[:, 0] = cfg.writing_x

    super().__init__(cfg, env)

    # Per-env metric tensor MUST exist (shape [B]) before any reset, and must
    # never be a scalar. The command manager indexes it as metric[env_ids],
    # so a 0-dim (scalar) value raises "too many indices for tensor of
    # dimension 0". Initialise it here as a proper per-environment tensor.
    self.metrics["pen_dist_to_target"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["pen_surface_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["pen_penetration"] = torch.zeros(self.num_envs, device=self.device)

    # Initialise with a random first sample so episode 0 is not all zeros.
    self._resample_command(torch.arange(self.num_envs, device=self.device))

  # ------------------------------------------------------------------
  # CommandTerm interface
  # ------------------------------------------------------------------

  @property
  def command(self) -> torch.Tensor:
    """Return target positions, shape [B, 3]."""
    return self._target

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._phase[env_ids] = 0.0
    self._set_circle_target(env_ids)

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    del env_ids
    self._phase += self._env.step_dt * self.cfg.angular_speed
    self._set_circle_target()

  def _set_circle_target(self, env_ids: torch.Tensor | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    phase = self._phase[ids]
    self._target[ids, 0] = self.cfg.writing_x
    self._target[ids, 1] = self.cfg.center_y + self.cfg.radius * torch.cos(phase)
    self._target[ids, 2] = self.cfg.center_z + self.cfg.radius * torch.sin(phase)

  def _update_metrics(self) -> None:
    pen_pos = self._get_pen_tip_pos()
    if pen_pos is not None:
      # Keep this PER-ENV (shape [B]); do NOT call .mean() — the command
      # manager indexes the metric as metric[env_ids] on reset.
      self.metrics["pen_dist_to_target"] = torch.norm(pen_pos - self._target, dim=1)
      self.metrics["pen_surface_error"] = torch.abs(
        pen_pos[:, 0] - self.cfg.writing_x
      )
      self.metrics["pen_penetration"] = torch.relu(pen_pos[:, 0] - BOARD_FACE_X)
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
  """Configuration for :class:`DrawTargetCommand`.

  Attributes:
      board_face_x: World X coordinate of the physical board face.
      writing_x: World X coordinate of the writing plane in front of the face.
      board_y_range: Valid horizontal range on the board.
      board_z_range: Valid vertical range on the board.
  """

  board_face_x: float = BOARD_FACE_X
  writing_x: float = WRITING_X
  board_y_range: tuple[float, float] = TARGET_Y_RANGE
  board_z_range: tuple[float, float] = TARGET_Z_RANGE
  center_y: float = BOARD_CENTRE_Y
  center_z: float = BOARD_CENTRE_Z
  radius: float = 0.15
  angular_speed: float = 0.25

  def build(self, env: ManagerBasedRlEnv) -> DrawTargetCommand:
    return DrawTargetCommand(self, env)
