"""drawing_canvas.py — Live whiteboard drawing for the G1 mjlab environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

# TYPE_CHECKING is False at runtime, so this import never executes and mujoco
# does not need to be installed just to import this module.  At analysis time
# Pylance sees the real MjModel / MjData / MjrContext types, so all the typed
# mujoco API calls below satisfy the type checker.
if TYPE_CHECKING:
    import mujoco

# Runtime availability flag — used to guard every mujoco call.
try:
    import mujoco as _mujoco_rt
    _MUJOCO_AVAILABLE = True
except ImportError:
    _mujoco_rt = None  # type: ignore[assignment]
    _MUJOCO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Board geometry — imported from the single source of truth in board.py rather
# than redefined here. These used to be a private copy that could (and did)
# drift out of step with the ranges used by the command and reward terms.
# ---------------------------------------------------------------------------

from .board import (  # noqa: E402
    BOARD_FACE_X,
    BOARD_Y_MAX,
    BOARD_Y_MIN,
    BOARD_Z_MAX,
    BOARD_Z_MIN,
    CONTACT_X_THRESHOLD,
)

PIXELS_PER_METRE: int = 300
BRUSH_RADIUS_PX: int = 4
INK_COLOUR: tuple[int, int, int] = (30, 30, 200)


class DrawingCanvas:
    """2-D numpy canvas that paints marks when the pen tip touches the board.

    Args:
        model:        ``mujoco.MjModel`` after the env is built, or ``None``
                      for canvas-only mode (unit tests).
        pen_tip_site: Site name in the MJCF.
        board_geom:   Geom name of the whiteboard surface in the MJCF.
        background:   RGB fill colour of a blank board.
    """

    def __init__(
        self,
        model: mujoco.MjModel | None,
        pen_tip_site: str = "pen_tip",
        board_geom: str = "whiteboard_surface",
        background: tuple[int, int, int] = (247, 247, 247),
    ) -> None:
        self._background = background

        y_span = BOARD_Y_MAX - BOARD_Y_MIN
        z_span = BOARD_Z_MAX - BOARD_Z_MIN
        self._canvas_w = int(round(y_span * PIXELS_PER_METRE))
        self._canvas_h = int(round(z_span * PIXELS_PER_METRE))

        self._canvas = np.full(
            (self._canvas_h, self._canvas_w, 3),
            fill_value=background,
            dtype=np.uint8,
        )

        self._pen_tip_id: int = -1
        self._texture_id: int = -1

        if model is not None and _MUJOCO_AVAILABLE and _mujoco_rt is not None:
            try:
                self._pen_tip_id = _mujoco_rt.mj_name2id(
                    model, _mujoco_rt.mjtObj.mjOBJ_SITE, pen_tip_site
                )
            except Exception:
                pass

            try:
                geom_id = _mujoco_rt.mj_name2id(
                    model, _mujoco_rt.mjtObj.mjOBJ_GEOM, board_geom
                )
                matid: int = int(model.geom_matid[geom_id])
                if matid >= 0:
                    # mat_texid has one column per texture ROLE in MuJoCo 3.x.
                    # The standard colour texture is the RGB role, NOT column 0
                    # (column 0 is mjTEXROLE_USER and is typically -1).  Read the
                    # RGB role; fall back to the first non-negative entry so this
                    # also works on older single-column layouts.
                    row = np.atleast_1d(model.mat_texid[matid])
                    rgb_role = int(_mujoco_rt.mjtTextureRole.mjTEXROLE_RGB)
                    tid = -1
                    if rgb_role < row.shape[0] and int(row[rgb_role]) >= 0:
                        tid = int(row[rgb_role])
                    else:
                        nonneg = [int(v) for v in row.tolist() if int(v) >= 0]
                        if nonneg:
                            tid = nonneg[0]
                    self._texture_id = tid
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def canvas(self) -> np.ndarray:
        return self._canvas

    def reset(self) -> None:
        """Clear all marks (call on episode reset for a fresh board)."""
        self._canvas[:] = self._background

    def step(self, data: mujoco.MjData) -> bool:
        """Paint a dot if the pen tip is touching the board this step.

        Args:
            data: Live ``mujoco.MjData``.

        Returns:
            ``True`` if a dot was painted, ``False`` otherwise.
        """
        if self._pen_tip_id < 0 or not _MUJOCO_AVAILABLE:
            return False

        pen_world = data.site_xpos[self._pen_tip_id]
        x = float(pen_world[0])
        y = float(pen_world[1])
        z = float(pen_world[2])

        if abs(x - BOARD_FACE_X) > CONTACT_X_THRESHOLD:
            return False
        if not (BOARD_Y_MIN <= y <= BOARD_Y_MAX and BOARD_Z_MIN <= z <= BOARD_Z_MAX):
            return False

        px = int((y - BOARD_Y_MIN) / (BOARD_Y_MAX - BOARD_Y_MIN) * (self._canvas_w - 1))
        py = int((1.0 - (z - BOARD_Z_MIN) / (BOARD_Z_MAX - BOARD_Z_MIN)) * (self._canvas_h - 1))
        self._paint_dot(px, py)
        return True

    def upload_texture(
        self,
        model: mujoco.MjModel,
        context: mujoco.MjrContext,
    ) -> bool:
        """Push the canvas into the MuJoCo whiteboard texture.

        Call after ``step()`` and before ``mjr_render``.

        Returns:
            ``True`` on success.
        """
        if self._texture_id < 0 or not _MUJOCO_AVAILABLE or _mujoco_rt is None:
            return False

        try:
            tex = model.tex(self._texture_id)
            # tex.width / tex.height may be 0-d or 1-element arrays depending on
            # the MuJoCo version; coerce to plain ints.
            target_h: int = int(np.atleast_1d(tex.height)[0])
            target_w: int = int(np.atleast_1d(tex.width)[0])
            if target_h <= 0 or target_w <= 0:
                return False

            pixels = (
                self._resize_canvas(target_h, target_w)
                if (self._canvas_h, self._canvas_w) != (target_h, target_w)
                else self._canvas
            )

            # tex.data is shaped (H, W, 3) uint8 in MuJoCo 3.x (the model.tex()
            # accessor reshapes it); on flat-buffer versions it is (H*W*3,).
            # Write whichever shape matches so we never broadcast-error.
            dst = tex.data
            pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
            if dst.ndim == 3:
                dst[:] = pixels
            else:
                dst[:] = pixels.reshape(-1)
            _mujoco_rt.mjr_uploadTexture(model, context, self._texture_id)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resize_canvas(self, target_h: int, target_w: int) -> np.ndarray:
        try:
            from PIL import Image
            try:
                resample = Image.Resampling.BILINEAR
            except AttributeError:
                resample = Image.LINEAR  # type: ignore[attr-defined]  # Pillow < 9.1
            img = Image.fromarray(self._canvas, "RGB")
            img = img.resize((target_w, target_h), resample)
            return np.asarray(img, dtype=np.uint8)
        except ImportError:
            y_idx = (np.arange(target_h) * self._canvas_h / target_h).astype(int)
            x_idx = (np.arange(target_w) * self._canvas_w / target_w).astype(int)
            return self._canvas[np.ix_(y_idx, x_idx)]

    def _paint_dot(self, cx: int, cy: int) -> None:
        r = BRUSH_RADIUS_PX
        y0 = max(0, cy - r)
        y1 = min(self._canvas_h - 1, cy + r)
        x0 = max(0, cx - r)
        x1 = min(self._canvas_w - 1, cx + r)
        ys, xs = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
        mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= r * r
        self._canvas[y0 : y1 + 1, x0 : x1 + 1][mask] = INK_COLOUR