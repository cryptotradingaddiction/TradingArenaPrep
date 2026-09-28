"""Event loop: synthetic feed → strategy → exchange → metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bot.strategy import build_strategy
from sim.exchange import Exchange, ExchangeConfig
from sim.feed import FeedConfig, SyntheticFeed


def _imbalance_hint_from_shock(shock: float, sigma: float, noise: float) -> float:
    """Map upcoming mid shock to a noisy book-size skew in [-1, 1]."""
    scale = max(sigma, 1e-6)
    raw = shock / (2.5 * scale) + noise
    return float(np.clip(raw, -1.0, 1.0))


def run(
    *,
    strategy_name: str = "mm",
    steps: int = 1000,
    seed: int = 0,
    tick: float = 0.01,
) -> dict:
    feed = SyntheticFeed(FeedConfig(seed=seed, tick=tick))
    ex = Exchange(ExchangeConfig(seed=seed + 1, tick=tick))
    strat = build_strategy(strategy_name)
    rng = np.random.default_rng(seed + 7)

    if strategy_name == "directional":
        max_inv = 25.0
        base_size = 4.0
    else:
        # mm and hybrid share inventory scale
        max_inv = 30.0
        base_size = 5.0

    cfg = {
        "tick": tick,
        "initial_cash": ex.cfg.initial_cash,
        "gamma": 0.1,
        "k_spread": 1.5,
        "kappa": 0.05,
        "alpha": 0.5,
        "base_size": base_size,
        "max_inventory": max_inv,
        "max_loss": 500.0,
        "imb_threshold": 0.15,
        "strong_threshold": 0.45,
    }
    strat.on_start(cfg)

    for _ in range(steps):
        shock = feed.peek_shock()
        # Predictive book skew for directional + hybrid; MM stays honest (hint=0)
        if strategy_name == "mm":
            hint = 0.0
        else:
            noise = float(rng.normal(0.0, 0.15))
            hint = _imbalance_hint_from_shock(shock, feed.cfg.sigma, noise)
        ex.seed_background(feed.mid, imbalance_hint=hint)

        top = ex.book.top(depth=5, ts=feed.t)
        if top is None:
            feed.step()
            continue

        strat.register_open([])
        reqs = strat.on_book(top, feed.t)
        fills = ex.apply_requests(reqs, feed.t)
        for f in fills:
            strat.on_fill(f)
        strat.register_open(ex.open_order_ids())
        for f in ex.maybe_external_taker(feed.t):
            strat.on_fill(f)
        strat.sync_state(ex.pos.qty, ex.pos.cash)

        mid = feed.step()
        ex.mark(mid)

    strat.on_end()
    summary = ex.metrics.summary()
    summary["realized_pnl"] = ex.pos.realized_pnl
    summary["cash"] = ex.pos.cash
    summary["qty"] = ex.pos.qty
    summary["mark_mid"] = feed.mid
    summary["strategy"] = strategy_name
    summary["max_inventory"] = max_inv
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ArenaPrep LOB simulator")
    p.add_argument(
        "--strategy",
        default="mm",
        choices=["mm", "directional", "dir", "imbalance", "hybrid"],
    )
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tick", type=float, default=0.01)
    args = p.parse_args(argv)

    name = "directional" if args.strategy in {"dir", "imbalance"} else args.strategy
    summary = run(
        strategy_name=name,
        steps=args.steps,
        seed=args.seed,
        tick=args.tick,
    )
    print(f"=== ArenaPrep {summary.get('strategy', name)} run ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
