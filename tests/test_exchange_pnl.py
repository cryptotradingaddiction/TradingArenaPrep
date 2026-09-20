"""Unit tests for apply_trade PnL accounting."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bot.types import Position
from sim.exchange import apply_trade


class TestApplyTrade(unittest.TestCase):
    def test_open_long_and_flatten(self):
        pos = Position(cash=1000.0)
        apply_trade(pos, +10.0, 100.0)  # buy 10 @ 100
        self.assertAlmostEqual(pos.qty, 10.0)
        self.assertAlmostEqual(pos.cash, 0.0)
        self.assertAlmostEqual(pos.avg_entry, 100.0)
        apply_trade(pos, -10.0, 110.0)  # sell 10 @ 110
        self.assertAlmostEqual(pos.qty, 0.0)
        self.assertAlmostEqual(pos.realized_pnl, 100.0)  # 10 * 10
        self.assertAlmostEqual(pos.cash, 1100.0)

    def test_open_short_and_flatten(self):
        pos = Position(cash=1000.0)
        apply_trade(pos, -5.0, 200.0)  # sell short
        self.assertAlmostEqual(pos.qty, -5.0)
        self.assertAlmostEqual(pos.avg_entry, 200.0)
        apply_trade(pos, +5.0, 180.0)  # cover cheaper
        self.assertAlmostEqual(pos.qty, 0.0)
        self.assertAlmostEqual(pos.realized_pnl, 100.0)  # 5 * 20

    def test_flip_long_to_short(self):
        pos = Position(cash=0.0)
        apply_trade(pos, +4.0, 50.0)
        apply_trade(pos, -6.0, 60.0)  # close 4, open short 2 @ 60
        self.assertAlmostEqual(pos.qty, -2.0)
        self.assertAlmostEqual(pos.avg_entry, 60.0)
        self.assertAlmostEqual(pos.realized_pnl, 40.0)  # 4 * 10

    def test_increase_long_avg(self):
        pos = Position()
        apply_trade(pos, +2.0, 100.0)
        apply_trade(pos, +2.0, 120.0)
        self.assertAlmostEqual(pos.qty, 4.0)
        self.assertAlmostEqual(pos.avg_entry, 110.0)


if __name__ == "__main__":
    unittest.main()
