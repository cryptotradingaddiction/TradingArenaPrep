"""Multi-seed PnL comparison: mm vs directional vs hybrid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sim.runner import run

STRATEGIES = ("mm", "directional", "hybrid")


def parse_seeds(spec: str) -> list[int]:
    """Parse '0-19' or '0,1,5' into a list of ints."""
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        lo, hi = int(a), int(b)
        if hi < lo:
            lo, hi = hi, lo
        return list(range(lo, hi + 1))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def benchmark(
    *,
    seeds: list[int],
    steps: int = 500,
    fee_bps: float = 1.0,
    adverse_half_ticks: float = 1.0,
    flatten_steps: int = 50,
    quote_every: int = 1,
) -> dict[str, dict[str, float]]:
    """Run all strategies on each seed once; return aggregate stats."""
    cache: dict[tuple[str, int], dict] = {}
    for name in STRATEGIES:
        for seed in seeds:
            cache[(name, seed)] = run(
                strategy_name=name,
                steps=steps,
                seed=seed,
                fee_bps=fee_bps,
                adverse_half_ticks=adverse_half_ticks,
                flatten_steps=flatten_steps,
                quote_every=quote_every,
            )

    rows: dict[str, dict[str, float]] = {}
    mm_eqs = [float(cache[("mm", s)]["final_equity"]) for s in seeds]
    for name in STRATEGIES:
        eqs = [float(cache[(name, s)]["final_equity"]) for s in seeds]
        dds = [float(cache[(name, s)]["max_drawdown"]) for s in seeds]
        fees = [float(cache[(name, s)].get("fees_paid", 0.0)) for s in seeds]
        arr = np.asarray(eqs, dtype=float)
        if name == "mm":
            win_rate = 1.0
        else:
            win_rate = sum(1 for a, b in zip(eqs, mm_eqs) if a >= b) / max(1, len(seeds))
        rows[name] = {
            "mean_equity": float(np.mean(arr)),
            "std_equity": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "mean_max_dd": float(np.mean(dds)),
            "mean_fees": float(np.mean(fees)),
            "win_rate_vs_mm": float(win_rate),
            "edge_vs_mm": float(np.mean(arr) - float(np.mean(mm_eqs))),
            "n": float(len(seeds)),
        }
    return rows


def format_table(rows: dict[str, dict[str, float]]) -> str:
    header = (
        f"{'strategy':<12} {'mean_eq':>10} {'std_eq':>10} "
        f"{'mean_dd':>10} {'fees':>10} {'win_vs_mm':>10} {'edge_vs_mm':>11}"
    )
    lines = [header, "-" * len(header)]
    for name in STRATEGIES:
        r = rows[name]
        lines.append(
            f"{name:<12} {r['mean_equity']:10.2f} {r['std_equity']:10.2f} "
            f"{r['mean_max_dd']:10.2f} {r['mean_fees']:10.4f} "
            f"{r['win_rate_vs_mm']:10.1%} {r['edge_vs_mm']:11.2f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ArenaPrep multi-seed strategy benchmark")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seeds", type=str, default="0-19", help="e.g. 0-19 or 0,1,2")
    p.add_argument("--fee-bps", type=float, default=1.0, dest="fee_bps")
    p.add_argument(
        "--adverse-half-ticks",
        type=float,
        default=1.0,
        dest="adverse_half_ticks",
    )
    p.add_argument("--flatten-steps", type=int, default=50, dest="flatten_steps")
    p.add_argument("--quote-every", type=int, default=1, dest="quote_every")
    args = p.parse_args(argv)

    seeds = parse_seeds(args.seeds)
    rows = benchmark(
        seeds=seeds,
        steps=args.steps,
        fee_bps=args.fee_bps,
        adverse_half_ticks=args.adverse_half_ticks,
        flatten_steps=args.flatten_steps,
        quote_every=args.quote_every,
    )
    print(
        f"=== ArenaPrep benchmark steps={args.steps} "
        f"seeds={seeds[0]}..{seeds[-1]} n={len(seeds)} ==="
    )
    print(format_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
