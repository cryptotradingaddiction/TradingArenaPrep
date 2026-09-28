"""Unit + integration tests for hybrid MM + imbalance skew."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bot.hybrid_mm_dir import HybridConfig, HybridMMDirStrategy
from bot.types import BookTop, Level, OrderAction, Side
from sim.runner import run


def _top(bid_sz: float, ask_sz: float, mid: float = 100.0) -> BookTop:
    tick = 0.01
    return BookTop(
        bid=mid - 3 * tick,
        ask=mid + 3 * tick,
        bid_size=bid_sz,
        ask_size=ask_sz,
        mid=mid,
        bids=(Level(mid - 3 * tick, bid_sz),),
        asks=(Level(mid + 3 * tick, ask_sz),),
        ts=0.0,
    )


class TestHybrid(unittest.TestCase):
    def test_flat_buy_imbalance_raises_reservation(self):
        h = HybridMMDirStrategy(HybridConfig(kappa=0.05, gamma=0.1))
        h.on_start({})
        mid, vol = 100.0, 0.05
        r0 = h.reservation_price(mid, inventory=0.0, vol=vol, imb_eff=0.0)
        r_buy = h.reservation_price(mid, inventory=0.0, vol=vol, imb_eff=0.8)
        self.assertAlmostEqual(r0, mid)
        self.assertGreater(r_buy, mid)

    def test_long_inventory_prefers_selling_despite_buy_imbalance(self):
        h = HybridMMDirStrategy(
            HybridConfig(gamma=0.1, kappa=0.05, alpha=0.5, max_inventory=30.0, base_size=5.0)
        )
        h.on_start({})
        h.inventory = 24.0  # heavily long → imb_eff faded
        for i in range(8):
            h._mids.append(100.0 + 0.01 * ((-1) ** i))
        # Strong buy pressure on the book
        top = _top(90, 10, mid=100.0)
        reqs = h.on_book(top, ts=1.0)
        places = [r for r in reqs if r.action is OrderAction.PLACE]
        bids = [r for r in places if r.side is Side.BUY]
        asks = [r for r in places if r.side is Side.SELL]
        self.assertTrue(bids and asks)  # still two-sided
        self.assertGreaterEqual(asks[0].size, bids[0].size)
        self.assertTrue(all(r.client_tag and r.client_tag.startswith("hybrid_") for r in places))

    def test_hybrid_integration_caps_and_equity(self):
        summary = run(strategy_name="hybrid", steps=200, seed=17)
        self.assertEqual(summary["strategy"], "hybrid")
        self.assertLessEqual(abs(summary["qty"]), summary["max_inventory"] + 1e-6)
        eq = summary["cash"] + summary["qty"] * summary["mark_mid"]
        self.assertTrue(math.isfinite(summary["final_equity"]))
        self.assertAlmostEqual(summary["final_equity"], eq, places=6)


if __name__ == "__main__":
    unittest.main()
