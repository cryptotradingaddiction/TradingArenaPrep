"""Unit tests for price-time LOB matching."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bot.types import Order, Side
from sim.book import OrderBook


class TestOrderBook(unittest.TestCase):
    def test_resting_bid_ask_top(self):
        book = OrderBook(tick=0.01)
        book.add(Order("b1", Side.BUY, 99.95, 10, 10, ts=1.0))
        book.add(Order("a1", Side.SELL, 100.05, 8, 8, ts=1.0))
        top = book.top()
        self.assertIsNotNone(top)
        assert top is not None
        self.assertAlmostEqual(top.bid, 99.95)
        self.assertAlmostEqual(top.ask, 100.05)
        self.assertAlmostEqual(top.mid, 100.0)

    def test_crossing_limit_buy_fills(self):
        book = OrderBook(tick=0.01)
        book.add(Order("a1", Side.SELL, 100.0, 5, 5, ts=0.0))
        fills = book.add(Order("b1", Side.BUY, 100.0, 3, 3, ts=1.0))
        self.assertEqual(len(fills), 1)
        self.assertAlmostEqual(fills[0].size, 3.0)
        self.assertAlmostEqual(fills[0].price, 100.0)
        # maker still has 2 remaining
        o = book.get_order("a1")
        self.assertIsNotNone(o)
        assert o is not None
        self.assertAlmostEqual(o.remaining, 2.0)

    def test_cancel_removes_order(self):
        book = OrderBook(tick=0.01)
        book.add(Order("b1", Side.BUY, 99.0, 1, 1))
        self.assertTrue(book.cancel("b1"))
        self.assertIsNone(book.get_order("b1"))
        self.assertFalse(book.cancel("b1"))

    def test_aggressive_hit_sweeps_asks(self):
        book = OrderBook(tick=0.01)
        book.add(Order("a1", Side.SELL, 100.0, 2, 2, client_tag="mm_ask"))
        book.add(Order("a2", Side.SELL, 100.01, 2, 2, client_tag="mm_ask"))
        fills = book.aggressive_hit(Side.BUY, 3.0, ts=2.0)
        self.assertAlmostEqual(sum(f.size for f in fills), 3.0)
        self.assertIsNone(book.get_order("a1"))
        o2 = book.get_order("a2")
        self.assertIsNotNone(o2)
        assert o2 is not None
        self.assertAlmostEqual(o2.remaining, 1.0)

    def test_fifo_within_price(self):
        book = OrderBook(tick=0.01)
        book.add(Order("a1", Side.SELL, 100.0, 1, 1, ts=1.0))
        book.add(Order("a2", Side.SELL, 100.0, 1, 1, ts=2.0))
        fills = book.aggressive_hit(Side.BUY, 1.0, ts=3.0)
        self.assertEqual(fills[0].order_id, "a1")


if __name__ == "__main__":
    unittest.main()
