"""DrawTargetCommand — issues random 3-D target positions on the whiteboard.

The whiteboard in the MJCF lives at world position (0.65, 0, 1.1) with a
half-extent of (0.02, 0.4, 0.35) in (x, y, z).  The face the robot writes on
is the -X face, so all targets share the constant X value of the board face
(0.65 - 0.02 = 0.63).  Y and Z are sampled uniformly within the board bounds.

Usage in env cfg::

    commands: dict[str, CommandTermCfg] = {
        "draw_target": DrawTargetCommandCfg(
            resampling_time_range=(3.0, 6.0),
            board_face_x=0.63,
            board_y_range=(-0.35, 0.35),
            board_z_range=(0.80, 1.40),
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

from .cfg_utils import resolve_first_site_id

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class DrawTargetCommand(CommandTerm):
    """Samples a random 3-D point on the whiteboard face each episode.

    The command stays fixed for ``resampling_time_range`` seconds, then a new
    point is drawn.  This gives the policy time to reach each target before it
    moves — important early in training.
    """

    cfg: DrawTargetCommandCfg

    def __init__(self, cfg: DrawTargetCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        # [B, 3]  —  (x=board_face_x, y=uniform, z=uniform)
        self._target = torch.zeros(self.num_envs, 3, device=self.device)
        self._target[:, 0] = cfg.board_face_x
        # Initialise with a random first sample so episode 0 is not all zeros.
        self._resample_command(torch.arange(self.num_envs, device=self.device))

        # Per-env metric tensor MUST exist (shape [B]) before any reset, and must
        # never be a scalar. The command manager indexes it as metric[env_ids],
        # so a 0-dim (scalar) value raises "too many indices for tensor of
        # dimension 0". Initialise it here as a proper per-environment tensor.
        self.metrics["pen_dist_to_target"] = torch.zeros(
            self.num_envs, device=self.device
        )

    # ------------------------------------------------------------------
    # CommandTerm interface
    # ------------------------------------------------------------------

    @property
    def command(self) -> torch.Tensor:
        """Return target positions, shape [B, 3]."""
        return self._target

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        self._target[env_ids, 1] = torch.empty(n, device=self.device).uniform_(
            *self.cfg.board_y_range
        )
        self._target[env_ids, 2] = torch.empty(n, device=self.device).uniform_(
            *self.cfg.board_z_range
        )

    def _update_command(self) -> None:
        pass

    def _update_metrics(self) -> None:
        pen_pos = self._get_pen_tip_pos()
        if pen_pos is not None:
            # Keep this PER-ENV (shape [B]); do NOT call .mean() — the command
            # manager indexes the metric as metric[env_ids] on reset.
            self.metrics["pen_dist_to_target"] = torch.norm(
                pen_pos - self._target, dim=1
            )
        # If pen_pos is None (early setup), keep the existing per-env tensor
        # already created in __init__ (shape [B]).

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_pen_tip_pos(self) -> torch.Tensor | None:
        """Try to read pen_tip site position from the robot.

        Returns ``None`` on any failure so ``_update_metrics`` degrades
        gracefully rather than crashing during early env setup.
        """
        try:
            asset_cfg = SceneEntityCfg("robot", site_names=("pen_tip",))
            asset_cfg.resolve(self._env.scene)
            # resolve_first_site_id handles list[int] | slice — no slice[0] issue.
            idx = resolve_first_site_id(asset_cfg)
            asset = self._env.scene["robot"]
            pos_w = asset.data.site_pos_w[:, idx, :]      # absolute world
            # Subtract per-env tiled offset so this matches the env-local target.
            return pos_w - self._env.scene.env_origins    # env-local
        except Exception:
            return None


@dataclass(kw_only=True)
class DrawTargetCommandCfg(CommandTermCfg):
    """Configuration for :class:`DrawTargetCommand`.

    Attributes:
        board_face_x:  World X coordinate of the board face (robot writes here).
                       Default matches the MJCF: board centre 0.65 − half-depth 0.02.
        board_y_range: (min, max) for Y sampling — horizontal on the board.
        board_z_range: (min, max) for Z sampling — vertical on the board.
                       Board Z spans 1.1 ± 0.35 = [0.75, 1.45]; we clip inside.
    """

    board_face_x: float = 0.63
    board_y_range: tuple[float, float] = (-0.30, 0.30)
    board_z_range: tuple[float, float] = (0.80, 1.40)

    def build(self, env: ManagerBasedRlEnv) -> DrawTargetCommand:
        return DrawTargetCommand(self, env)