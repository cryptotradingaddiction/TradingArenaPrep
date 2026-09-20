"""Edge-case tests for OrderBook (hangs, crossed markets)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bot.types import Order, Side
from sim.book import OrderBook


class TestBookEdge(unittest.TestCase):
    def test_empty_level_aggressive_hit_does_not_hang(self):
        book = OrderBook(tick=0.01)
        book.add(Order("a1", Side.SELL, 100.0, 1, 1, client_tag="mm_ask"))
        # Inject a stale empty price key (would hang without purge guard)
        _ = book._asks[99.99]  # defaultdict creates empty deque
        fills = book.aggressive_hit(Side.BUY, 10.0, ts=1.0)
        self.assertGreaterEqual(sum(f.size for f in fills), 1.0)
        self.assertNotIn(99.99, book._asks)

    def test_crossed_book_top_returns_none(self):
        book = OrderBook(tick=0.01)
        book.add(Order("b1", Side.BUY, 100.05, 1, 1))
        book.add(Order("a1", Side.SELL, 100.00, 1, 1))
        # Force crossed by resting without match (bypass add matching via direct rest)
        book2 = OrderBook(tick=0.01)
        book2._rest(Order("b1", Side.BUY, 100.10, 1, 1))
        book2._rest(Order("a1", Side.SELL, 100.00, 1, 1))
        self.assertIsNone(book2.top())

    def test_match_skips_empty_ask_level(self):
        book = OrderBook(tick=0.01)
        book._asks[100.0]  # empty key
        book.add(Order("a1", Side.SELL, 100.01, 2, 2))
        fills = book.add(Order("b1", Side.BUY, 100.01, 2, 2))
        self.assertAlmostEqual(sum(f.size for f in fills), 2.0)


if __name__ == "__main__":
    unittest.main()
