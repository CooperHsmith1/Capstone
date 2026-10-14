"""Convert an image to pen-plotter G-code (tone contour line art) for the whiteboard.

Usage:
  uv run python examples/image_to_gcode.py input.png examples/portrait.gcode \
      [--width 100] [--max-points 1500] [--levels 3]

The image is blurred and iso-tone contours are traced at a few grey levels.
Each contour is one pen-down stroke; travel between strokes is pen-up.
"""

import argparse

import numpy as np
from contourpy import contour_generator
from PIL import Image, ImageFilter


def trace(img: Image.Image, size: int, levels: int, min_len: float, blur: float):
  img = img.convert("L")
  h = round(img.height * size / img.width)
  img = img.resize((size, h), Image.LANCZOS).filter(ImageFilter.GaussianBlur(blur))
  a = np.asarray(img, dtype=np.float64)
  gen = contour_generator(z=a)
  strokes = []
  for lv in np.linspace(a.min(), a.max(), levels + 2)[1:-1]:
    for line in gen.lines(lv):
      length = np.sum(np.hypot(*np.diff(line, axis=0).T))
      if length >= min_len:
        strokes.append(line)
  return strokes, h


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("image")
  p.add_argument("output")
  p.add_argument("--width", type=float, default=100.0, help="drawing width in mm")
  p.add_argument("--max-points", type=int, default=2500)
  p.add_argument("--levels", type=int, default=5)
  args = p.parse_args()

  img = Image.open(args.image)
  min_len = 6.0
  while True:
    strokes, h = trace(img, 160, args.levels, min_len, blur=1.2)
    n = sum(len(s[::3]) for s in strokes)
    if n <= args.max_points or min_len > 80:
      break
    min_len *= 1.25
  strokes.sort(key=len, reverse=True)

  scale = args.width / 160
  lines = ["G21 ; millimetres", "G90 ; absolute", "G0 Z5"]
  for s in strokes:
    pts = np.vstack([s[::3], s[-1:]])
    lines += [f"G0 X{pts[0][0] * scale:.2f} Y{(h - pts[0][1]) * scale:.2f}", "G1 Z0"]
    lines += [f"G1 X{x * scale:.2f} Y{(h - y) * scale:.2f}" for x, y in pts[1:]]
    lines.append("G0 Z5")
  with open(args.output, "w") as f:
    f.write("\n".join(lines) + "\n")
  print(f"{len(strokes)} strokes, {n} points -> {args.output}")


if __name__ == "__main__":
  main()
