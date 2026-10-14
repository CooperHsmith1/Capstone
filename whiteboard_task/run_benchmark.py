#!/usr/bin/env python3
"""Entry point for the comparative filter benchmark (report Section 4.3).

    python run_benchmark.py

Equivalent to ``python -m mdp.state_estimation.benchmark`` but without the
runpy double-import warning that appears because the package ``__init__``
already imports the benchmark module. Needs only torch -- no mjlab, so this
runs on a laptop while the simulator is training elsewhere.
"""

import argparse
import sys

import torch
from mdp.state_estimation.benchmark import format_table, run_benchmark


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--steps", type=int, default=1000)
  parser.add_argument("--envs", type=int, default=8)
  parser.add_argument("--particles", type=int, default=512)
  parser.add_argument(
    "--repeats",
    type=int,
    default=5,
    help="Independent noise realisations per cell. More gives tighter "
    "error bars; anything below 3 is not enough to separate a real "
    "difference from run-to-run variance.",
  )
  parser.add_argument("--sigma", type=float, default=0.01, help="Sensor noise std (m).")
  parser.add_argument("--device", default="cpu")
  args = parser.parse_args()

  torch.manual_seed(0)
  results = run_benchmark(
    steps=args.steps,
    num_envs=args.envs,
    num_particles=args.particles,
    repeats=args.repeats,
    sigma=args.sigma,
    device=args.device,
  )
  print(format_table(results))
  print()
  print("Interpretation:")
  print("  'Improve' is the fraction of measurement error the filter removes.")
  print("  A negative value means the filter is worse than the raw sensor.")
  print("  '*' marks the lowest mean RMSE in each regime.")
  print("  Differences smaller than the +/- spread are not real differences.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
