"""Integration: full runner respects caps and equity identity."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bot.mm_inventory import MMConfig, MMInventoryStrategy
from sim.runner import run


class TestIntegration(unittest.TestCase):
    def test_mm_run_inventory_and_equity(self):
        summary = run(strategy_name="mm", steps=200, seed=11)
        max_inv = summary["max_inventory"]
        self.assertLessEqual(abs(summary["qty"]), max_inv + 1e-6)
        eq = summary["cash"] + summary["qty"] * summary["mark_mid"]
        self.assertTrue(math.isfinite(summary["final_equity"]))
        self.assertAlmostEqual(summary["final_equity"], eq, places=6)

    def test_directional_run_inventory_and_equity(self):
        summary = run(strategy_name="directional", steps=200, seed=11)
        max_inv = summary["max_inventory"]
        self.assertLessEqual(abs(summary["qty"]), max_inv + 1e-6)
        eq = summary["cash"] + summary["qty"] * summary["mark_mid"]
        self.assertAlmostEqual(summary["final_equity"], eq, places=6)

    def test_mm_room_clamp_near_cap(self):
        mm = MMInventoryStrategy(MMConfig(max_inventory=10.0, base_size=5.0))
        mm.on_start({})
        mm.inventory = 8.0
        bid, ask = mm._sizes()
        self.assertLessEqual(bid, 2.0 + 1e-9)
        self.assertGreater(ask, 0.0)


if __name__ == "__main__":
    unittest.main()
