"""Continuous Gerono figure-eight target command for whiteboard drawing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
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
from .gcode import GcodePath, load_gcode_path

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

INK_COLOR = (0.1, 0.2, 1.0, 1.0)


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
    self._pen_down = torch.ones(num_envs, dtype=torch.bool, device=device)

    # Ink trace: env-local (y, z) of every pen-on-board mark, as a ring buffer.
    self._ink = torch.zeros(num_envs, cfg.ink_max_points, 2, device=device)
    self._ink_count = torch.zeros(num_envs, dtype=torch.long, device=device)
    self._ink_last = torch.full((num_envs, 2), float("inf"), device=device)

    # G-code mode: a dense board-space path and a per-env float path index.
    self._path_yz: torch.Tensor | None = None
    self._path_down: torch.Tensor | None = None
    self._path_s = torch.zeros(num_envs, device=device)
    self._path_spacing = 0.005
    if cfg.gcode_path is not None:
      path = load_gcode_path(cfg.gcode_path, cfg.gcode_max_size)
      self._set_path(path, device)

    super().__init__(cfg, env)

    # Per-env metric tensor MUST exist (shape [B]) before any reset, and must
    # never be a scalar. The command manager indexes it as metric[env_ids],
    # so a 0-dim (scalar) value raises "too many indices for tensor of
    # dimension 0". Initialise it here as a proper per-environment tensor.
    self.metrics["pen_dist_to_target"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["path_progress"] = torch.zeros(self.num_envs, device=self.device)

    # Start at the center target so the arm can reach before drawing begins.
    self._resample_command(torch.arange(self.num_envs, device=self.device))

  # ------------------------------------------------------------------
  # CommandTerm interface
  # ------------------------------------------------------------------

  @property
  def command(self) -> torch.Tensor:
    """Return target positions, shape [B, 3]."""
    return self._target

  @property
  def pen_down(self) -> torch.Tensor:
    """Whether the pen should currently touch the board, shape [B] bool."""
    return self._pen_down

  @property
  def progress(self) -> torch.Tensor:
    """Fraction of the G-code path completed, shape [B]; 0 without a path."""
    if self._path_yz is None:
      return torch.zeros_like(self._elapsed)
    return self._path_s / (len(self._path_yz) - 1)

  @property
  def path_length_m(self) -> float:
    """Total G-code path length in metres (0 without a path)."""
    if self._path_yz is None:
      return 0.0
    return self._path_spacing * (len(self._path_yz) - 1)

  def _set_path(self, path: GcodePath, device: str | torch.device) -> None:
    self._path_yz = torch.as_tensor(path.points, dtype=torch.float32, device=device)
    self._path_down = torch.as_tensor(path.pen_down, dtype=torch.bool, device=device)
    self._path_spacing = path.spacing

  @property
  def ink_points(self) -> torch.Tensor:
    """Ink marks as env-local (y, z), shape [B, N, 2]; valid rows < ink_count."""
    return self._ink

  @property
  def ink_count(self) -> torch.Tensor:
    """Total marks left per env (the ring buffer keeps the newest N), [B]."""
    return self._ink_count

  def _record_ink(self, ids: torch.Tensor | slice) -> None:
    pen_pos = self._get_pen_tip_pos()
    if pen_pos is None:
      return
    try:
      touching = self._env.scene[self.cfg.contact_sensor_name].data.found.any(dim=1)
    except KeyError:
      return
    yz = pen_pos[:, 1:]
    far = torch.norm(yz - self._ink_last, dim=1) >= self.cfg.ink_min_spacing
    mask = touching & far
    if isinstance(ids, slice):
      sel = torch.nonzero(mask).flatten()
    else:
      sel = ids[mask[ids]]
    if sel.numel() == 0:
      return
    slot = self._ink_count[sel] % self.cfg.ink_max_points
    self._ink[sel, slot] = yz[sel]
    self._ink_last[sel] = yz[sel]
    self._ink_count[sel] += 1

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    origins = self._env.scene.env_origins
    for i in visualizer.get_env_indices(self.num_envs):
      n = min(int(self._ink_count[i]), self.cfg.ink_max_points)
      pts = self._ink[i, :n].cpu().numpy()
      origin = origins[i].cpu().numpy()
      for y, z in pts:
        center = np.array(
          [origin[0] + self.cfg.writing_x, origin[1] + y, origin[2] + z],
          dtype=np.float64,
        )
        visualizer.add_sphere(center, self.cfg.ink_radius, INK_COLOR)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._ink_count[env_ids] = 0
    self._ink_last[env_ids] = float("inf")
    self._elapsed[env_ids] = 0.0
    self._phase[env_ids] = 0.0
    self._path_s[env_ids] = 0.0
    self._set_target(env_ids)

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self._elapsed[ids] += self._env.step_dt
    self._record_ink(ids)
    if self._path_yz is not None:
      self._advance_path(ids)
      self._set_target(env_ids)
      return
    self._phase[ids] = (self._elapsed[ids] - self.cfg.approach_duration_s).clamp_min(
      0.0
    ) * self.cfg.angular_speed
    self._set_figure_eight_target(env_ids)

  def _set_target(self, env_ids: torch.Tensor | None = None) -> None:
    if self._path_yz is not None:
      self._set_path_target(env_ids)
    else:
      self._set_figure_eight_target(env_ids)

  def _advance_path(self, ids: torch.Tensor | slice) -> None:
    """Advance along the path only while the robot is stable and on target.

    Progress is gated so that nothing is "drawn" until the robot is upright,
    the pen is on the current waypoint and, for pen-down waypoints, the tip is
    actually touching the board. Otherwise the target waits for the policy.
    """
    pen_pos = self._get_pen_tip_pos()
    if pen_pos is None:
      return
    ready = self._elapsed[ids] >= self.cfg.approach_duration_s
    ready = ready & (
      torch.norm(pen_pos[ids] - self._target[ids], dim=1) < self.cfg.advance_tolerance
    )
    robot = self._env.scene["robot"]
    tilt = robot.data.projected_gravity_b[ids, :2].square().sum(dim=1)
    ready = ready & (tilt < self.cfg.max_tilt_sq)
    try:
      touching = self._env.scene[self.cfg.contact_sensor_name].data.found
      touching = touching.any(dim=1)[ids]
    except KeyError:
      touching = torch.ones_like(ready)
    ready = ready & (touching | ~self._pen_down[ids])
    step = self.cfg.path_speed * self._env.step_dt / self._path_spacing
    last = float(len(self._path_yz) - 1)  # type: ignore[arg-type]
    self._path_s[ids] = (self._path_s[ids] + step * ready.float()).clamp_max(last)

  def _set_path_target(self, env_ids: torch.Tensor | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    assert self._path_yz is not None and self._path_down is not None
    s = self._path_s[ids]
    lo = s.floor().long()
    hi = (lo + 1).clamp_max(len(self._path_yz) - 1)
    frac = (s - lo.float()).unsqueeze(-1)
    yz = self._path_yz[lo] * (1.0 - frac) + self._path_yz[hi] * frac
    down = self._path_down[hi]
    self._pen_down[ids] = down
    self._target[ids, 0] = self.cfg.writing_x - self.cfg.pen_lift * (~down).float()
    self._target[ids, 1:] = yz

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
    self.metrics["path_progress"] = self.progress
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
  gcode_path: str | None = None
  """G-code file to draw instead of the figure eight."""
  gcode_max_size: tuple[float, float] = (0.40, 0.40)
  """Max (width, height) in metres the G-code drawing is scaled to fit."""
  path_speed: float = 0.05
  """Pen speed along the G-code path in m/s while gating conditions hold."""
  advance_tolerance: float = 0.04
  """Pen-to-waypoint distance (m) within which the path may advance."""
  max_tilt_sq: float = 0.1
  """Max squared sideways gravity component for the path to advance."""
  pen_lift: float = 0.03
  """Hover distance (m) off the board for pen-up travel moves."""
  contact_sensor_name: str = "pen_board_contact"
  ink_max_points: int = 2000
  """Ring-buffer size of the blue ink trace left on the board, per env."""
  ink_min_spacing: float = 0.004
  """Minimum distance (m) between consecutive ink marks."""
  ink_radius: float = 0.004
  """Rendered radius (m) of each ink mark."""

  def build(self, env: ManagerBasedRlEnv) -> DrawTargetCommand:
    return DrawTargetCommand(self, env)
