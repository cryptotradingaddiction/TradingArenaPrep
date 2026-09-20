"""Unit tests for MM inventory skew and caps."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bot.mm_inventory import MMConfig, MMInventoryStrategy
from bot.types import BookTop, Fill, OrderAction, Side


class TestMMInventory(unittest.TestCase):
    def test_long_inventory_lowers_reservation(self):
        mm = MMInventoryStrategy(MMConfig(gamma=0.2, tick=0.01))
        mid, vol = 100.0, 0.05
        r_flat = mm.reservation_price(mid, inventory=0.0, vol=vol)
        r_long = mm.reservation_price(mid, inventory=20.0, vol=vol)
        r_short = mm.reservation_price(mid, inventory=-20.0, vol=vol)
        self.assertAlmostEqual(r_flat, mid)
        self.assertLess(r_long, mid)   # prefer selling when long
        self.assertGreater(r_short, mid)

    def test_quotes_skew_when_long(self):
        mm = MMInventoryStrategy(MMConfig(gamma=0.5, k_spread=1.0, tick=0.01, base_size=5.0))
        mm.on_start({"initial_cash": 10_000.0})
        # Seed vol window
        top0 = BookTop(99.9, 100.1, 10, 10, 100.0, ts=0.0)
        for i in range(10):
            mm.on_book(
                BookTop(99.9, 100.1, 10, 10, 100.0 + 0.01 * ((-1) ** i), ts=float(i)),
                ts=float(i),
            )
        mm.inventory = 25.0
        reqs = mm.on_book(top0, ts=20.0)
        places = [r for r in reqs if r.action is OrderAction.PLACE]
        bids = [r for r in places if r.side is Side.BUY]
        asks = [r for r in places if r.side is Side.SELL]
        self.assertTrue(bids and asks)
        # reservation below mid → both quotes shifted down vs unskewed mid±half
        self.assertLess(bids[0].price, top0.mid)
        self.assertLess(asks[0].price, top0.mid + 1.0)

    def test_inventory_cap_blocks_bid(self):
        mm = MMInventoryStrategy(MMConfig(max_inventory=10.0, base_size=5.0))
        mm.on_start({})
        mm.inventory = 10.0
        for i in range(5):
            mm._mids.append(100.0 + 0.01 * i)
        top = BookTop(99.95, 100.05, 10, 10, 100.0, ts=1.0)
        reqs = mm.on_book(top, ts=1.0)
        places = [r for r in reqs if r.action is OrderAction.PLACE]
        self.assertFalse(any(r.side is Side.BUY for r in places))
        self.assertTrue(any(r.side is Side.SELL for r in places))

    def test_sizes_clamp_to_remaining_room(self):
        mm = MMInventoryStrategy(MMConfig(max_inventory=10.0, base_size=5.0))
        mm.inventory = 9.0
        bid, ask = mm._sizes()
        # taper: 5 * (1 - 9/10) = 0.5, room_buy = 1 → min = 0.5
        self.assertAlmostEqual(bid, 0.5)
        self.assertGreater(ask, 0.0)
        mm.inventory = 9.5
        bid2, _ = mm._sizes()
        # taper would be 0.25 but room is 0.5 → still 0.25; force room bind:
        mm.cfg.base_size = 10.0
        bid3, _ = mm._sizes()
        self.assertAlmostEqual(bid3, 0.5)  # room_buy = 0.5 wins

    def test_on_fill_updates_inventory(self):
        mm = MMInventoryStrategy()
        mm.on_start({})
        mm.on_fill(Fill("o1", Side.BUY, 100.0, 3.0, ts=1.0, client_tag="mm_bid"))
        self.assertAlmostEqual(mm.inventory, 3.0)
        mm.on_fill(Fill("o2", Side.SELL, 100.2, 1.0, ts=2.0, client_tag="mm_ask"))
        self.assertAlmostEqual(mm.inventory, 2.0)


if __name__ == "__main__":
    unittest.main()
