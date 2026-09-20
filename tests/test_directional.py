"""Unit tests for directional imbalance strategy."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bot.directional_imbalance import (
    DirectionalConfig,
    DirectionalImbalanceStrategy,
    imbalance,
)
from bot.types import BookTop, Level, OrderAction, Side


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


class TestDirectional(unittest.TestCase):
    def test_imbalance_sign(self):
        self.assertGreater(imbalance(_top(80, 20)), 0.0)
        self.assertLess(imbalance(_top(20, 80)), 0.0)
        self.assertAlmostEqual(imbalance(_top(50, 50)), 0.0)

    def test_strong_buy_pressure_takes_ask(self):
        strat = DirectionalImbalanceStrategy(
            DirectionalConfig(imb_threshold=0.1, strong_threshold=0.4, base_size=2.0)
        )
        strat.on_start({})
        reqs = strat.on_book(_top(90, 10), ts=1.0)
        places = [r for r in reqs if r.action is OrderAction.PLACE]
        self.assertEqual(len(places), 1)
        self.assertIs(places[0].side, Side.BUY)
        self.assertEqual(places[0].client_tag, "dir_take_buy")

    def test_mild_sell_pressure_improves_ask(self):
        strat = DirectionalImbalanceStrategy(
            DirectionalConfig(imb_threshold=0.1, strong_threshold=0.9, base_size=2.0)
        )
        strat.on_start({})
        # imb = (30-70)/100 = -0.4 → mild short lean
        top = _top(30, 70)
        reqs = strat.on_book(top, ts=1.0)
        places = [r for r in reqs if r.action is OrderAction.PLACE]
        self.assertEqual(len(places), 1)
        self.assertIs(places[0].side, Side.SELL)
        self.assertEqual(places[0].client_tag, "dir_ask")
        self.assertAlmostEqual(places[0].price, top.ask - 0.01)

    def test_flat_when_balanced(self):
        strat = DirectionalImbalanceStrategy(DirectionalConfig(imb_threshold=0.2))
        strat.on_start({})
        reqs = strat.on_book(_top(52, 48), ts=1.0)
        places = [r for r in reqs if r.action is OrderAction.PLACE]
        self.assertEqual(places, [])

    def test_inventory_cap_blocks_long(self):
        strat = DirectionalImbalanceStrategy(
            DirectionalConfig(max_inventory=5.0, strong_threshold=0.2)
        )
        strat.on_start({})
        strat.inventory = 5.0
        reqs = strat.on_book(_top(90, 10), ts=1.0)
        places = [r for r in reqs if r.action is OrderAction.PLACE]
        self.assertEqual(places, [])


if __name__ == "__main__":
    unittest.main()
