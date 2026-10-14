"""Generate a 3D-printable two-part marker clamp for the G1 right hand.

Usage:
  uv run python examples/make_marker_clamp.py [--marker-diameter 17.0]

The clamp matches the simulated marker (barrel diameter 17 mm, see PEN_RADIUS in
``mjlab/tasks/whiteboard/mdp/assets.py``). The base half has a tunnel for a velcro
strap that wraps the hand; the cap half bolts on with four M3 screws. Line the
channel with 1 mm of rubber or foam so the marker is gripped, not just held.
Always check the fit against the real hand before relying on it.
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh


def box(extents, centre) -> trimesh.Trimesh:
  m = trimesh.creation.box(extents=extents)
  m.apply_translation(centre)
  return m


def cylinder_along(axis: str, radius: float, length: float, centre) -> trimesh.Trimesh:
  m = trimesh.creation.cylinder(radius=radius, height=length, sections=64)
  rotations = {"x": [0, 1, 0], "y": [1, 0, 0], "z": [0, 0, 1]}
  if axis != "z":
    angle = np.pi / 2
    m.apply_transform(trimesh.transformations.rotation_matrix(angle, rotations[axis]))
  m.apply_translation(centre)
  return m


def build(marker_d: float, clearance: float = 0.4) -> tuple[trimesh.Trimesh, ...]:
  length, width, h_cap, h_base = 60.0, 34.0, 12.0, 15.0
  r = marker_d / 2 + clearance
  channel = cylinder_along("x", r, length + 2, (0, 0, 0))
  bolts = [
    cylinder_along("z", 1.7, 40.0, (x, y, 0))
    for x in (-20.0, 20.0)
    for y in (-(r + 4.2), r + 4.2)
  ]
  # Strap tunnel runs across the base, under the marker channel.
  tunnel = box((22.0, width + 2, 3.2), (0, 0, -(h_base - 3.0)))

  base = box((length, width, h_base), (0, 0, -h_base / 2))
  base = base.difference(channel).difference(tunnel)
  for b in bolts:
    base = base.difference(b)

  cap = box((length, width, h_cap), (0, 0, h_cap / 2))
  cap = cap.difference(channel)
  for b in bolts:
    cap = cap.difference(b)
  return base, cap


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--marker-diameter", type=float, default=17.0, help="mm")
  p.add_argument("--out", default="examples/marker_clamp")
  args = p.parse_args()

  base, cap = build(args.marker_diameter)
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  cap.apply_transform(  # print the cap flat-side down
    trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])
  )
  cap.apply_translation([0, 0, -cap.bounds[0][2]])
  base.apply_translation([0, 0, -base.bounds[0][2]])
  for name, mesh in (("clamp_base", base), ("clamp_cap", cap)):
    assert mesh.is_watertight, f"{name} is not watertight"
    mesh.export(out / f"{name}.stl")
    print(
      f"{name}: extents {np.round(mesh.extents, 1)} mm, volume {mesh.volume:.0f} mm3"
    )


if __name__ == "__main__":
  main()
