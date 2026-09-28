"""End-of-round flatten and quote_every / benchmark smoke."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sim.benchmark import benchmark, parse_seeds
from sim.runner import run


class TestFlatten(unittest.TestCase):
    def test_flatten_leaves_flat_inventory(self):
        for name in ("mm", "directional", "hybrid"):
            s = run(
                strategy_name=name,
                steps=120,
                seed=3,
                flatten_steps=40,
                fee_bps=1.0,
                adverse_half_ticks=1.0,
            )
            self.assertLess(
                abs(s["qty"]),
                1e-6,
                msg=f"{name} leftover qty={s['qty']}",
            )

    def test_quote_every_completes(self):
        s = run(strategy_name="hybrid", steps=80, seed=2, quote_every=5, flatten_steps=10)
        self.assertIn("final_equity", s)
        self.assertEqual(s["quote_every"], 5)


class TestBenchmarkSmoke(unittest.TestCase):
    def test_parse_seeds(self):
        self.assertEqual(parse_seeds("0-3"), [0, 1, 2, 3])
        self.assertEqual(parse_seeds("1,4,7"), [1, 4, 7])

    def test_benchmark_two_seeds(self):
        rows = benchmark(seeds=[0, 1], steps=40, flatten_steps=10)
        for name in ("mm", "directional", "hybrid"):
            self.assertIn(name, rows)
            self.assertIn("mean_equity", rows[name])
            self.assertIn("win_rate_vs_mm", rows[name])


if __name__ == "__main__":
    unittest.main()
