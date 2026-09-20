"""Market making with inventory skew (simplified Avellaneda–Stoikov)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List

from bot.types import (
    BookTop,
    Fill,
    OrderAction,
    OrderRequest,
    Side,
)


@dataclass
class MMConfig:
    tick: float = 0.01
    gamma: float = 0.1          # inventory risk aversion
    k_spread: float = 1.5       # half-spread = max(tick, k_spread * vol)
    base_size: float = 5.0
    max_inventory: float = 30.0
    max_loss: float = 500.0
    vol_window: int = 50
    min_vol: float = 0.01
    quote_every: int = 1        # replace quotes every N book updates
    initial_cash: float = 10_000.0


class MMInventoryStrategy:
    """Two-sided quotes around reservation price skewed by inventory."""

    def __init__(self, config: MMConfig | None = None) -> None:
        self.cfg = config or MMConfig()
        self.inventory = 0.0
        self.cash = self.cfg.initial_cash
        self._mids: Deque[float] = deque(maxlen=max(5, self.cfg.vol_window))
        self._step = 0
        self._open_ids: List[str] = []
        self._start_equity = self.cfg.initial_cash
        self._stopped = False

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

    def sync_state(self, qty: float, cash: float) -> None:
        """Overwrite local ledger from exchange (source of truth)."""
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
        half = max(self.cfg.tick, self.cfg.k_spread * vol)
        reservation = top.mid - self.cfg.gamma * self.inventory * (vol ** 2)
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
            # Still crossed / locked — do not place
            return self._cancel_all()

        bid_size, ask_size = self._sizes()
        reqs = self._cancel_all()
        if bid_size > 0:
            reqs.append(
                OrderRequest(
                    action=OrderAction.PLACE,
                    side=Side.BUY,
                    price=bid_px,
                    size=bid_size,
                    client_tag="mm_bid",
                )
            )
        if ask_size > 0:
            reqs.append(
                OrderRequest(
                    action=OrderAction.PLACE,
                    side=Side.SELL,
                    price=ask_px,
                    size=ask_size,
                    client_tag="mm_ask",
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

    def reservation_price(self, mid: float, inventory: float, vol: float) -> float:
        """Exposed for unit tests: long inventory → lower reservation."""
        return mid - self.cfg.gamma * inventory * (vol ** 2)

    def register_open(self, order_ids: List[str]) -> None:
        self._open_ids = list(order_ids)

    def _sizes(self) -> tuple[float, float]:
        cap = self.cfg.max_inventory
        inv = self.inventory
        base = self.cfg.base_size
        room_buy = max(0.0, cap - inv)
        room_sell = max(0.0, cap + inv)
        bid_size = min(base * max(0.0, 1.0 - max(0.0, inv) / cap), room_buy)
        ask_size = min(base * max(0.0, 1.0 - max(0.0, -inv) / cap), room_sell)
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
