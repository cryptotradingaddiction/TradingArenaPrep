"""
Directional strategy: book imbalance → short-horizon mid lean.

Signal:
    imb = (bid_sz - ask_sz) / (bid_sz + ask_sz)   # in [-1, 1]

Actions:
    imb > +thr  → lean long  (improve bid / take ask if strong)
    imb < -thr  → lean short (improve ask / take bid if strong)
    else        → flat (cancel)

Inventory and max-loss caps mirror the MM bot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from bot.types import (
    BookTop,
    Fill,
    OrderAction,
    OrderRequest,
    OrderType,
    Side,
)


@dataclass
class DirectionalConfig:
    tick: float = 0.01
    imb_threshold: float = 0.15
    strong_threshold: float = 0.45   # above this → cross the spread (taker)
    base_size: float = 4.0
    max_inventory: float = 25.0
    max_loss: float = 500.0
    initial_cash: float = 10_000.0
    size_scale: float = 8.0          # size ≈ base + size_scale * |imb|


def imbalance(top: BookTop) -> float:
    """Book imbalance in [-1, 1]."""
    denom = top.bid_size + top.ask_size
    if denom <= 1e-12:
        return 0.0
    return (top.bid_size - top.ask_size) / denom


class DirectionalImbalanceStrategy:
    """One-sided / aggressive quotes driven by top-of-book imbalance."""

    def __init__(self, config: DirectionalConfig | None = None) -> None:
        self.cfg = config or DirectionalConfig()
        self.inventory = 0.0
        self.cash = self.cfg.initial_cash
        self._open_ids: List[str] = []
        self._start_equity = self.cfg.initial_cash
        self._stopped = False
        self.last_imb = 0.0

    def on_start(self, config: dict) -> None:
        for k, v in config.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
        self.cash = float(config.get("initial_cash", self.cfg.initial_cash))
        self._start_equity = self.cash
        self.inventory = 0.0
        self._open_ids.clear()
        self._stopped = False
        self.last_imb = 0.0

    def sync_state(self, qty: float, cash: float) -> None:
        """Overwrite local ledger from exchange (source of truth)."""
        self.inventory = float(qty)
        self.cash = float(cash)

    def on_book(self, top: BookTop, ts: float) -> List[OrderRequest]:
        _ = ts
        if self._stopped:
            return self._cancel_all()
        equity = self.cash + self.inventory * top.mid
        if equity < self._start_equity - self.cfg.max_loss:
            self._stopped = True
            return self._cancel_all()

        imb = imbalance(top)
        self.last_imb = imb
        reqs = self._cancel_all()
        thr = self.cfg.imb_threshold
        if abs(imb) < thr:
            return reqs

        if imb > 0:
            room = self.cfg.max_inventory - self.inventory
        else:
            room = self.cfg.max_inventory + self.inventory
        if room <= 1e-12:
            return reqs

        size = self.cfg.base_size + self.cfg.size_scale * abs(imb)
        size = min(size, room)
        if size <= 1e-12:
            return reqs

        strong = abs(imb) >= self.cfg.strong_threshold
        tick = self.cfg.tick

        if imb > 0:
            if strong:
                reqs.append(
                    OrderRequest(
                        action=OrderAction.PLACE,
                        side=Side.BUY,
                        price=top.ask,
                        size=size,
                        order_type=OrderType.LIMIT,
                        client_tag="dir_take_buy",
                    )
                )
            else:
                px = self._round(top.bid + tick)
                if px >= top.ask - 1e-12:
                    return reqs  # would cross — skip improve
                reqs.append(
                    OrderRequest(
                        action=OrderAction.PLACE,
                        side=Side.BUY,
                        price=px,
                        size=size,
                        client_tag="dir_bid",
                    )
                )
        else:
            if strong:
                reqs.append(
                    OrderRequest(
                        action=OrderAction.PLACE,
                        side=Side.SELL,
                        price=top.bid,
                        size=size,
                        order_type=OrderType.LIMIT,
                        client_tag="dir_take_sell",
                    )
                )
            else:
                px = self._round(top.ask - tick)
                if px <= top.bid + 1e-12:
                    return reqs
                reqs.append(
                    OrderRequest(
                        action=OrderAction.PLACE,
                        side=Side.SELL,
                        price=px,
                        size=size,
                        client_tag="dir_ask",
                    )
                )
        return reqs

    def on_fill(self, fill: Fill) -> None:
        if fill.side is Side.BUY:
            self.inventory += fill.size
            self.cash -= fill.price * fill.size
        else:
            self.inventory -= fill.size
            self.cash += fill.price * fill.size

    def on_end(self) -> None:
        pass

    def register_open(self, order_ids: List[str]) -> None:
        self._open_ids = list(order_ids)

    def _round(self, px: float) -> float:
        t = self.cfg.tick
        return round(round(px / t) * t, 10)

    def _cancel_all(self) -> List[OrderRequest]:
        reqs = [
            OrderRequest(action=OrderAction.CANCEL, order_id=oid)
            for oid in self._open_ids
        ]
        self._open_ids.clear()
        return reqs
