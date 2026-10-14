"""G-code to whiteboard pen-path conversion.

Supports the subset of G-code produced by pen-plotter / laser toolpath
generators (Inkscape, vpype, LightBurn, ...):

* ``G0``/``G1`` linear moves, ``G2``/``G3`` arcs with ``I``/``J`` centre offsets.
* ``G90``/``G91`` absolute/relative and ``G20``/``G21`` inch/mm units.
* Pen state from ``Z`` (``Z <= 0`` is pen down), ``M3``/``M5`` or
  ``M300 S<angle>`` (``S < 40`` is pen down). ``G1`` moves with no pen command
  at all are treated as drawing and ``G0`` moves as travel.
* ``;`` and ``( )`` comments, line numbers and ``%`` delimiters.

G-code X maps to board Y (rightwards) and G-code Y maps to board Z (upwards).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .board import BOARD_CENTRE_Y, BOARD_CENTRE_Z

__all__ = ["GcodePath", "parse_gcode", "load_gcode_path", "fit_to_board"]

_WORD = re.compile(r"([A-Za-z])\s*([-+]?(?:\d+\.?\d*|\.\d+))")
_COMMENT = re.compile(r"\([^)]*\)|;.*")
_ARC_SEGMENT_RAD = math.radians(5.0)


@dataclass(frozen=True)
class GcodePath:
  """Dense pen path in board coordinates.

  Attributes:
    points: ``[P, 2]`` board (Y, Z) positions in metres.
    pen_down: ``[P]`` bool, True where the pen should touch the board.
    spacing: Arc-length distance between consecutive points in metres.
  """

  points: np.ndarray
  pen_down: np.ndarray
  spacing: float

  @property
  def length(self) -> float:
    return self.spacing * (len(self.points) - 1)


def _strip(line: str) -> str:
  return _COMMENT.sub("", line).strip().lstrip("%")


def _axis(
  args: dict[str, float], axis: str, current: float, scale: float, absolute: bool
) -> float:
  if axis not in args:
    return current
  value = args[axis] * scale
  return value if absolute else current + value


def parse_gcode(text: str) -> list[tuple[float, float, bool]]:
  """Parse G-code text into raw ``(x, y, pen_down)`` vertices in millimetres."""
  x = y = 0.0
  z = 1.0
  absolute = True
  scale = 1.0
  pen_down: bool | None = None  # None: infer from the motion command
  vertices: list[tuple[float, float, bool]] = []

  for raw in text.splitlines():
    line = _strip(raw)
    if not line:
      continue
    words = [(m.group(1).upper(), float(m.group(2))) for m in _WORD.finditer(line)]
    codes = {(letter, value) for letter, value in words}
    args = {letter: value for letter, value in words if letter not in "GMN"}

    if ("G", 90.0) in codes:
      absolute = True
    if ("G", 91.0) in codes:
      absolute = False
    if ("G", 20.0) in codes:
      scale = 25.4
    if ("G", 21.0) in codes:
      scale = 1.0
    if ("M", 3.0) in codes or ("M", 4.0) in codes:
      pen_down = True
    if ("M", 5.0) in codes:
      pen_down = False
    if ("M", 300.0) in codes and "S" in args:
      pen_down = args["S"] < 40.0

    motion = next(
      (int(v) for k, v in words if k == "G" and int(v) in (0, 1, 2, 3)), None
    )
    if motion is None and not any(a in args for a in "XYZ"):
      continue
    if motion is None:
      motion = 1

    nx = _axis(args, "X", x, scale, absolute)
    ny = _axis(args, "Y", y, scale, absolute)
    nz = _axis(args, "Z", z, scale, absolute)
    if "Z" in args and not any(c in args for c in "XY"):
      pen_down = nz <= 0.0
      z = nz
      continue
    if "Z" in args:
      pen_down = nz <= 0.0
    z = nz

    down = pen_down if pen_down is not None else motion != 0
    if not vertices:
      vertices.append((x, y, False))
    if motion in (2, 3) and ("I" in args or "J" in args):
      cx = x + args.get("I", 0.0) * scale
      cy = y + args.get("J", 0.0) * scale
      a0 = math.atan2(y - cy, x - cx)
      a1 = math.atan2(ny - cy, nx - cx)
      sweep = a1 - a0
      if motion == 2:  # clockwise
        if sweep >= 0.0:
          sweep -= 2.0 * math.pi
      elif sweep <= 0.0:
        sweep += 2.0 * math.pi
      if math.isclose(x, nx) and math.isclose(y, ny):
        sweep = -2.0 * math.pi if motion == 2 else 2.0 * math.pi
      radius = math.hypot(x - cx, y - cy)
      steps = max(2, int(abs(sweep) / _ARC_SEGMENT_RAD))
      for k in range(1, steps + 1):
        a = a0 + sweep * k / steps
        vertices.append((cx + radius * math.cos(a), cy + radius * math.sin(a), down))
    else:
      vertices.append((nx, ny, down))
    x, y = nx, ny

  return vertices


def _resample(
  vertices: list[tuple[float, float, bool]], spacing_mm: float
) -> tuple[np.ndarray, np.ndarray]:
  """Resample a polyline to uniform arc length. Segment pen state follows its end."""
  pts = np.array([(v[0], v[1]) for v in vertices], dtype=np.float64)
  down = np.array([v[2] for v in vertices], dtype=bool)
  seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
  keep = np.concatenate([[True], seg > 1e-9])
  pts, down = pts[keep], down[keep]
  seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
  cum = np.concatenate([[0.0], np.cumsum(seg)])
  n = max(2, int(math.ceil(cum[-1] / spacing_mm)) + 1)
  s = np.linspace(0.0, cum[-1], n)
  idx = np.clip(np.searchsorted(cum, s, side="left"), 1, len(pts) - 1)
  out = np.stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])], axis=1)
  return out, down[idx]


def fit_to_board(
  vertices: list[tuple[float, float, bool]],
  max_size: tuple[float, float] = (0.40, 0.40),
  spacing: float = 0.005,
  center: tuple[float, float] = (BOARD_CENTRE_Y, BOARD_CENTRE_Z),
) -> GcodePath:
  """Scale (uniformly) and centre raw G-code vertices to fit the writing area.

  Leading and trailing pen-up travel is trimmed so the path starts and ends on
  the board surface.
  """
  down_idx = [i for i, v in enumerate(vertices) if v[2]]
  if len(down_idx) < 2:
    raise ValueError("G-code contains no pen-down (drawing) moves.")
  vertices = vertices[down_idx[0] - 1 if down_idx[0] > 0 else 0 : down_idx[-1] + 1]
  pts, down = _resample(vertices, spacing_mm=1.0)

  drawn = pts[down] if down.any() else pts
  lo, hi = drawn.min(axis=0), drawn.max(axis=0)
  extent = np.maximum(hi - lo, 1e-6)
  scale = min(max_size[0] / extent[0], max_size[1] / extent[1])
  mid = (lo + hi) / 2.0
  pts_m = (pts - mid) * scale + np.array(center)
  # Resample again in metres so spacing is uniform after scaling.
  dense, dense_down = _resample(
    [(p[0], p[1], bool(d)) for p, d in zip(pts_m, down, strict=True)],
    spacing_mm=spacing,
  )
  first = int(np.argmax(dense_down))
  return GcodePath(points=dense[first:], pen_down=dense_down[first:], spacing=spacing)


def load_gcode_path(
  path: str | Path,
  max_size: tuple[float, float] = (0.40, 0.40),
  spacing: float = 0.005,
) -> GcodePath:
  """Read a G-code file and convert it to a board-fitted :class:`GcodePath`."""
  text = Path(path).expanduser().read_text(encoding="utf-8", errors="replace")
  return fit_to_board(parse_gcode(text), max_size=max_size, spacing=spacing)
