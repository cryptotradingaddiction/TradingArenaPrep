"""Tests for fees and maker-hit adverse selection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bot.types import Fill, Order, OrderAction, OrderRequest, Side
from sim.exchange import Exchange, ExchangeConfig


class TestExchangeFees(unittest.TestCase):
    def test_strategy_taker_fill_pays_fee(self):
        ex = Exchange(ExchangeConfig(fee_bps=10.0, adverse_half_ticks=0.0, seed=0))
        ex.seed_background(100.0)
        cash0 = ex.pos.cash
        # Cross the ask (buy as taker)
        top = ex.book.top()
        assert top is not None
        fills = ex.apply_requests(
            [
                OrderRequest(
                    action=OrderAction.PLACE,
                    side=Side.BUY,
                    price=top.ask,
                    size=2.0,
                    client_tag="mm_bid",
                )
            ],
            ts=1.0,
        )
        self.assertTrue(fills)
        notional = sum(f.price * f.size for f in fills)
        expected_fee = 10.0 / 1e4 * notional
        # cash fell by notional + fee
        self.assertAlmostEqual(ex.metrics.fees_paid, expected_fee, places=6)
        self.assertLess(ex.pos.cash, cash0 - notional + 1e-9)

    def test_maker_hit_buy_worsens_by_half_tick(self):
        tick = 0.01
        ex = Exchange(
            ExchangeConfig(
                fee_bps=0.0,
                adverse_half_ticks=1.0,
                tick=tick,
                taker_intensity=1.0,
                taker_size_mean=5.0,
                seed=1,
            )
        )
        ex.seed_background(100.0)
        # Rest a bid inside the spread
        top = ex.book.top()
        assert top is not None
        bid_px = ex.book.round_price(top.bid + tick)
        ex.apply_requests(
            [
                OrderRequest(
                    action=OrderAction.PLACE,
                    side=Side.BUY,
                    price=bid_px,
                    size=3.0,
                    client_tag="mm_bid",
                )
            ],
            ts=1.0,
        )
        # Force many external sells into our bid
        got = []
        for _ in range(40):
            got.extend(ex.maybe_external_taker(2.0))
            if got:
                break
        self.assertTrue(got)
        buy_fills = [f for f in got if f.side is Side.BUY]
        self.assertTrue(buy_fills)
        # Adverse: paid bid + 0.5 tick
        self.assertAlmostEqual(buy_fills[0].price, bid_px + 0.5 * tick, places=8)


if __name__ == "__main__":
    unittest.main()
