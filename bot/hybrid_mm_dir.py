"""
Hybrid strategy: two-sided MM quotes with imbalance skew (never pure taker).

    imb = (bid_sz - ask_sz) / (bid_sz + ask_sz)
    imb_eff = imb * (1 - |inv| / max_inventory)   # fade when loaded
    r = mid - gamma * inv * vol^2 + kappa * imb_eff * vol
    sizes tilted by alpha * imb_eff, then room-clamped
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List

from bot.directional_imbalance import imbalance
from bot.types import (
    BookTop,
    Fill,
    OrderAction,
    OrderRequest,
    Side,
)


@dataclass
class HybridConfig:
    tick: float = 0.01
    gamma: float = 0.1
    k_spread: float = 1.5
    kappa: float = 0.05       # imbalance reservation tilt
    alpha: float = 0.5        # imbalance size tilt
    base_size: float = 5.0
    max_inventory: float = 30.0
    max_loss: float = 500.0
    vol_window: int = 50
    min_vol: float = 0.01
    quote_every: int = 1
    initial_cash: float = 10_000.0


class HybridMMDirStrategy:
    """MM base + faded imbalance skew; always two-sided when room allows."""

    def __init__(self, config: HybridConfig | None = None) -> None:
        self.cfg = config or HybridConfig()
        self.inventory = 0.0
        self.cash = self.cfg.initial_cash
        self._mids: Deque[float] = deque(maxlen=max(5, self.cfg.vol_window))
        self._step = 0
        self._open_ids: List[str] = []
        self._start_equity = self.cfg.initial_cash
        self._stopped = False
        self.last_imb = 0.0
        self.last_imb_eff = 0.0

    def on_start(self, config: dict) -> None:
        for k, v in config.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
        self.cash = float(config.get("initial_cash", self.cfg.initial_cash))
        self._start_equity = self.cash
        self.inventory = 0.0
        self._mids.clear()
        self._step = 0
        self._open_ids.clear()
        self._stopped = False
        self.last_imb = 0.0
        self.last_imb_eff = 0.0

    def sync_state(self, qty: float, cash: float) -> None:
        self.inventory = float(qty)
        self.cash = float(cash)

    def on_book(self, top: BookTop, ts: float) -> List[OrderRequest]:
        self._step += 1
        self._mids.append(top.mid)
        if self._stopped:
            return self._cancel_all()
        equity = self.cash + self.inventory * top.mid
        if equity < self._start_equity - self.cfg.max_loss:
            self._stopped = True
            return self._cancel_all()
        if self.cfg.quote_every > 1 and (self._step % self.cfg.quote_every) != 0:
            return []

        vol = self._rolling_vol()
        imb = imbalance(top)
        imb_eff = self._imb_eff(imb)
        self.last_imb = imb
        self.last_imb_eff = imb_eff

        half = max(self.cfg.tick, self.cfg.k_spread * vol)
        reservation = self.reservation_price(top.mid, self.inventory, vol, imb_eff)
        bid_px = self._round(reservation - half)
        ask_px = self._round(reservation + half)
        improve_bid = self._round(top.bid + self.cfg.tick)
        improve_ask = self._round(top.ask - self.cfg.tick)
        bid_px = max(bid_px, improve_bid)
        ask_px = min(ask_px, improve_ask)
        if ask_px < bid_px + self.cfg.tick - 1e-12:
            bid_px = self._round(top.mid - self.cfg.tick)
            ask_px = self._round(top.mid + self.cfg.tick)
        if ask_px < bid_px + self.cfg.tick - 1e-12:
            return self._cancel_all()

        bid_size, ask_size = self._sizes(imb_eff)
        reqs = self._cancel_all()
        if bid_size > 0:
            reqs.append(
                OrderRequest(
                    action=OrderAction.PLACE,
                    side=Side.BUY,
                    price=bid_px,
                    size=bid_size,
                    client_tag="hybrid_bid",
                )
            )
        if ask_size > 0:
            reqs.append(
                OrderRequest(
                    action=OrderAction.PLACE,
                    side=Side.SELL,
                    price=ask_px,
                    size=ask_size,
                    client_tag="hybrid_ask",
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

    def reservation_price(
        self,
        mid: float,
        inventory: float,
        vol: float,
        imb_eff: float = 0.0,
    ) -> float:
        """mid - gamma*inv*vol^2 + kappa*imb_eff*vol."""
        return (
            mid
            - self.cfg.gamma * inventory * (vol ** 2)
            + self.cfg.kappa * imb_eff * vol
        )

    def register_open(self, order_ids: List[str]) -> None:
        self._open_ids = list(order_ids)

    def _imb_eff(self, imb: float) -> float:
        cap = max(self.cfg.max_inventory, 1e-12)
        fade = max(0.0, 1.0 - abs(self.inventory) / cap)
        return float(imb) * fade

    def _sizes(self, imb_eff: float) -> tuple[float, float]:
        cap = self.cfg.max_inventory
        inv = self.inventory
        base = self.cfg.base_size
        room_buy = max(0.0, cap - inv)
        room_sell = max(0.0, cap + inv)
        bid_size = base * max(0.0, 1.0 - max(0.0, inv) / cap)
        ask_size = base * max(0.0, 1.0 - max(0.0, -inv) / cap)
        # Imbalance tilt (still two-sided; no zeroing from signal alone)
        bid_size *= max(0.0, 1.0 + self.cfg.alpha * imb_eff)
        ask_size *= max(0.0, 1.0 - self.cfg.alpha * imb_eff)
        bid_size = min(bid_size, room_buy)
        ask_size = min(ask_size, room_sell)
        if room_buy <= 1e-12:
            bid_size = 0.0
        if room_sell <= 1e-12:
            ask_size = 0.0
        return bid_size, ask_size

    def _rolling_vol(self) -> float:
        if len(self._mids) < 3:
            return self.cfg.min_vol
        arr = list(self._mids)
        rets = [arr[i] - arr[i - 1] for i in range(1, len(arr))]
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / max(1, len(rets) - 1)
        return max(self.cfg.min_vol, var ** 0.5)

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
