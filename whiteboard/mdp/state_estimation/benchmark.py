"""Comparative benchmarking harness for the three filters (Section 4.3).

Runs EKF, UKF and the particle filter over identical measurement sequences and
reports accuracy against wall-clock cost. Deliberately depends only on torch so
it runs without mjlab -- you can generate the comparison table for the report on
any machine, including while the sim is training.

Run directly::

    python -m mdp.state_estimation.benchmark

The noise regimes matter more than the filter list. Under clean Gaussian noise
all three are near-optimal and score within a few percent of each other; the
particle filter's advantage only appears once the noise stops being Gaussian.
Reporting both regimes is the honest result -- a table showing only the
outlier case would overstate the particle filter, and one showing only the
Gaussian case would understate it.
"""

from __future__ import annotations

import time
import zlib
from dataclasses import dataclass

import torch

from .base import StateEstimator, StateEstimatorCfg
from .kalman import ExtendedKalmanFilter, UnscentedKalmanFilter
from .particle_filter import ParticleFilter


def regime_seed(name: str) -> int:
    """Stable seed derived from a regime name.

    ``hash()`` is deliberately avoided: CPython salts string hashing per
    process (PYTHONHASHSEED), so a hash-derived seed silently changes the
    noise realisation on every run.
    """
    return zlib.crc32(name.encode("utf-8")) % 100_000


@dataclass
class BenchmarkResult:
    """One filter's score under one noise regime."""

    filter_name: str
    regime: str
    rmse_m: float
    raw_rmse_m: float
    max_error_m: float
    wall_time_ms: float
    steps: int
    seed: int = 0

    @property
    def improvement(self) -> float:
        """Fraction of measurement error removed. Negative means it made it worse."""
        if self.raw_rmse_m <= 0.0:
            return 0.0
        return 1.0 - (self.rmse_m / self.raw_rmse_m)

    @property
    def ms_per_step(self) -> float:
        return self.wall_time_ms / max(self.steps, 1)


def generate_trajectory(
    steps: int = 1000,
    num_envs: int = 8,
    dt: float = 0.02,
    device: str = "cpu",
) -> torch.Tensor:
    """Smooth pen trajectory over the whiteboard face, shape [T, B, 3].

    A Lissajous sweep is used rather than a straight line because it has
    continuously varying acceleration, which is what actually stresses the
    constant-velocity motion model the filters assume.
    """
    t = torch.arange(steps, device=device, dtype=torch.float32) * dt
    t = t.unsqueeze(1).expand(steps, num_envs)
    phase = torch.linspace(0.0, 2.0, num_envs, device=device).unsqueeze(0)

    x = torch.full_like(t, 0.63) + 0.005 * torch.sin(4.0 * t + phase)
    y = 0.25 * torch.sin(1.5 * t + phase)
    z = 1.10 + 0.20 * torch.sin(2.3 * t + phase)
    return torch.stack([x, y, z], dim=-1)


def corrupt(
    truth: torch.Tensor,
    sigma: float,
    outlier_rate: float = 0.0,
    outlier_scale: float = 25.0,
    dropout_rate: float = 0.0,
    seed: int = 0,
) -> torch.Tensor:
    """Apply a noise regime to a ground-truth trajectory.

    Args:
        truth: [T, B, 3] ground truth.
        sigma: Gaussian noise std dev (m).
        outlier_rate: Per-step probability of a gross outlier.
        outlier_scale: Outlier size as a multiple of sigma.
        dropout_rate: Per-step probability the sensor holds its previous value
            (models a frame drop rather than a wrong reading).
        seed: RNG seed.
    """
    g = torch.Generator(device=truth.device)
    g.manual_seed(seed)

    noisy = truth + torch.randn(truth.shape, generator=g, device=truth.device) * sigma

    if outlier_rate > 0.0:
        mask = torch.rand(truth.shape[:2], generator=g, device=truth.device) < outlier_rate
        spike = torch.randn(truth.shape, generator=g, device=truth.device) * (sigma * outlier_scale)
        noisy = torch.where(mask.unsqueeze(-1), truth + spike, noisy)

    if dropout_rate > 0.0:
        drop = torch.rand(truth.shape[:2], generator=g, device=truth.device) < dropout_rate
        held = noisy.clone()
        for t in range(1, truth.shape[0]):
            held[t] = torch.where(drop[t].unsqueeze(-1), held[t - 1], noisy[t])
        noisy = held

    return noisy


def evaluate(
    estimator: StateEstimator,
    measurements: torch.Tensor,
    truth: torch.Tensor,
    dt: float,
    regime: str,
) -> BenchmarkResult:
    """Run one filter over one measurement sequence and score it."""
    estimator.reset(measurement=measurements[0])

    torch.cuda.synchronize() if measurements.is_cuda else None
    start = time.perf_counter()

    estimates = []
    for t in range(measurements.shape[0]):
        estimator.predict(dt)
        estimator.update(measurements[t])
        estimates.append(estimator.position.clone())

    torch.cuda.synchronize() if measurements.is_cuda else None
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    est = torch.stack(estimates)
    err = torch.linalg.norm(est - truth, dim=-1)
    raw_err = torch.linalg.norm(measurements - truth, dim=-1)

    return BenchmarkResult(
        filter_name=estimator.name,
        regime=regime,
        rmse_m=float(torch.sqrt(torch.mean(err**2))),
        raw_rmse_m=float(torch.sqrt(torch.mean(raw_err**2))),
        max_error_m=float(err.max()),
        wall_time_ms=elapsed_ms,
        steps=measurements.shape[0],
    )


def run_benchmark(
    steps: int = 1000,
    num_envs: int = 8,
    dt: float = 0.02,
    sigma: float = 0.01,
    num_particles: int = 512,
    repeats: int = 5,
    device: str = "cpu",
) -> list[BenchmarkResult]:
    """Run every filter against every noise regime.

    Returns:
        A flat list of results, one per (filter, regime) pair.
    """
    truth = generate_trajectory(steps, num_envs, dt, device)

    regimes = {
        "gaussian": dict(sigma=sigma),
        "outliers_6pct": dict(sigma=sigma, outlier_rate=0.06, outlier_scale=30.0),
        "dropout_15pct": dict(sigma=sigma, dropout_rate=0.15),
        "heavy_noise": dict(sigma=sigma * 4.0),
    }

    # One shared noise config so any difference is the algorithm, not tuning.
    cfg = StateEstimatorCfg(measurement_noise=sigma)
    B = num_envs

    results: list[BenchmarkResult] = []
    for regime_name, kwargs in regimes.items():
        for rep in range(repeats):
            # Deterministic per-(regime, repeat) seed. Do NOT use hash() here:
            # Python salts str hashing per process, so hash-derived seeds give
            # a different noise draw on every run and the table cannot be
            # reproduced or cited. crc32 is stable across processes and
            # machines.
            seed = (regime_seed(regime_name) + rep * 7919) % 2**31

            meas = corrupt(truth, seed=seed, **kwargs)

            builders = [
                lambda: ExtendedKalmanFilter(B, cfg, device=device),
                lambda: UnscentedKalmanFilter(B, cfg, device=device),
                lambda: ParticleFilter(
                    B, cfg, num_particles=num_particles,
                    proposal="bootstrap", device=device, seed=seed,
                ),
                lambda: ParticleFilter(
                    B, cfg, num_particles=num_particles,
                    proposal="optimal", device=device, seed=seed,
                ),
            ]
            labels = [
                "EKF",
                "UKF",
                f"PF-bootstrap(N={num_particles})",
                f"PF-optimal(N={num_particles})",
            ]

            for build, label in zip(builders, labels):
                est = build()
                res = evaluate(est, meas, truth, dt, regime_name)
                res.filter_name = label
                res.seed = seed
                results.append(res)

    return results


def aggregate(results: list[BenchmarkResult]) -> dict[tuple[str, str], dict[str, float]]:
    """Collapse repeated runs into mean and standard deviation per cell.

    A single run of this benchmark is one draw from a noisy process. Reporting
    one number per filter invites exactly the mistake this harness previously
    made -- concluding that a filter "wins" from what was really seed variance.
    Aggregating over repeats and quoting the spread is what makes the
    comparison defensible in the report.
    """
    buckets: dict[tuple[str, str], list[BenchmarkResult]] = {}
    for r in results:
        buckets.setdefault((r.regime, r.filter_name), []).append(r)

    out: dict[tuple[str, str], dict[str, float]] = {}
    for key, group in buckets.items():
        rmse = torch.tensor([g.rmse_m for g in group])
        raw = torch.tensor([g.raw_rmse_m for g in group])
        out[key] = {
            "rmse_mean": float(rmse.mean()),
            "rmse_std": float(rmse.std(unbiased=False)),
            "raw_mean": float(raw.mean()),
            "improvement": float(1.0 - rmse.mean() / raw.mean()),
            "max_error": max(g.max_error_m for g in group),
            "us_per_step": float(
                torch.tensor([g.ms_per_step for g in group]).mean() * 1000.0
            ),
            "n": len(group),
        }
    return out


def format_table(results: list[BenchmarkResult]) -> str:
    """Render aggregated results as a fixed-width table, grouped by regime.

    RMSE is reported as mean +/- std over the repeated noise realisations, so a
    difference between two filters can be read against the run-to-run spread
    rather than taken at face value.
    """
    agg = aggregate(results)

    regimes: list[str] = []
    filters: list[str] = []
    for r in results:
        if r.regime not in regimes:
            regimes.append(r.regime)
        if r.filter_name not in filters:
            filters.append(r.filter_name)

    n_reps = agg[(regimes[0], filters[0])]["n"]

    header = (
        f"{'Filter':<26} {'RMSE (mm)':>18} {'Raw (mm)':>10} "
        f"{'Improve':>9} {'Max err (mm)':>13} {'us/step':>10}"
    )

    lines: list[str] = [
        f"Aggregated over {n_reps} independent noise realisations "
        f"(mean +/- std).",
    ]

    for regime in regimes:
        lines.append("")
        lines.append(f"Noise regime: {regime}")
        lines.append("-" * len(header))
        lines.append(header)
        lines.append("-" * len(header))

        best = min(agg[(regime, f)]["rmse_mean"] for f in filters)
        for f in filters:
            a = agg[(regime, f)]
            marker = " *" if a["rmse_mean"] == best else "  "
            rmse_str = f"{a['rmse_mean']*1000:8.2f} +/- {a['rmse_std']*1000:5.2f}"
            lines.append(
                f"{f:<26} {rmse_str:>18} "
                f"{a['raw_mean']*1000:>10.2f} "
                f"{a['improvement']*100:>8.1f}% "
                f"{a['max_error']*1000:>13.2f} "
                f"{a['us_per_step']:>10.1f}{marker}"
            )

    return "\n".join(lines)


def main() -> None:
    torch.manual_seed(0)
    results = run_benchmark()
    print(format_table(results))
    print()
    print("Interpretation:")
    print("  'Improve' is the fraction of measurement error the filter removes.")
    print("  A negative value means the filter is worse than the raw sensor.")
    print("  '*' marks the lowest mean RMSE in each regime.")
    print("  Compare us/step against accuracy -- that trade-off is the point of")
    print("  the comparison, not the RMSE column on its own.")
    print("  Differences smaller than the +/- spread are not real differences.")


if __name__ == "__main__":
    main()
