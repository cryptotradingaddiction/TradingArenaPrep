"""Exchange wrapper: account, order ids, PnL, background liquidity + takers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from bot.types import (
    Fill,
    Order,
    OrderAction,
    OrderRequest,
    OrderType,
    Position,
    Side,
)
from sim.book import OrderBook


@dataclass
class ExchangeConfig:
    tick: float = 0.01
    initial_cash: float = 10_000.0
    depth_levels: int = 5
    background_size: float = 50.0
    taker_intensity: float = 0.35   # base prob of an aggressive hit each step
    taker_size_mean: float = 2.0
    seed: int = 0


@dataclass
class ExchangeMetrics:
    """Fill counters: ``fills_total`` = all book events; ``strategy_fills`` = ours only."""

    fills: int = 0
    strategy_fills: int = 0
    inventory_path: List[float] = field(default_factory=list)
    equity_path: List[float] = field(default_factory=list)
    spread_captured: List[float] = field(default_factory=list)
    max_drawdown: float = 0.0

    def summary(self) -> dict:
        peak = 0.0
        max_dd = 0.0
        for eq in self.equity_path:
            peak = max(peak, eq)
            max_dd = max(max_dd, peak - eq)
        self.max_drawdown = max_dd
        return {
            "fills_total": self.fills,
            "strategy_fills": self.strategy_fills,
            "final_inventory": self.inventory_path[-1] if self.inventory_path else 0.0,
            "final_equity": self.equity_path[-1] if self.equity_path else 0.0,
            "mean_spread_captured": (
                float(np.mean(self.spread_captured)) if self.spread_captured else 0.0
            ),
            "max_drawdown": self.max_drawdown,
        }


def apply_trade(pos: Position, qty_delta: float, price: float) -> None:
    """Update inventory, cash, avg_entry, and realized_pnl for one fill.

    ``qty_delta`` > 0 = buy, < 0 = sell. Covers open / increase / reduce / flatten / flip.
    """
    if abs(qty_delta) <= 1e-15:
        return
    price = float(price)
    qty_delta = float(qty_delta)
    # Cash: buy pays, sell receives
    pos.cash -= price * qty_delta

    old_qty = pos.qty
    new_qty = old_qty + qty_delta

    if abs(old_qty) <= 1e-15:
        # Flat → open
        pos.avg_entry = price
        pos.qty = new_qty
        return

    same_dir = (old_qty > 0 and qty_delta > 0) or (old_qty < 0 and qty_delta < 0)
    if same_dir:
        total = abs(old_qty) + abs(qty_delta)
        pos.avg_entry = (abs(old_qty) * pos.avg_entry + abs(qty_delta) * price) / total
        pos.qty = new_qty
        return

    # Reducing or flipping
    closed = min(abs(old_qty), abs(qty_delta))
    if old_qty > 0:
        # Closing long with sells
        pos.realized_pnl += (price - pos.avg_entry) * closed
    else:
        # Closing short with buys
        pos.realized_pnl += (pos.avg_entry - price) * closed

    if abs(new_qty) <= 1e-12:
        pos.qty = 0.0
        pos.avg_entry = 0.0
    elif (old_qty > 0 and new_qty < 0) or (old_qty < 0 and new_qty > 0):
        # Flip: residual opens at this fill price
        pos.qty = new_qty
        pos.avg_entry = price
    else:
        # Partial reduce, same sign remains
        pos.qty = new_qty


class Exchange:
    """Owns the LOB, strategy open orders, and mark-to-mid PnL (source of truth)."""

    def __init__(self, config: ExchangeConfig | None = None) -> None:
        self.cfg = config or ExchangeConfig()
        self.book = OrderBook(tick=self.cfg.tick)
        self.pos = Position(cash=self.cfg.initial_cash)
        self.metrics = ExchangeMetrics()
        self.rng = np.random.default_rng(self.cfg.seed)
        self._next_id = 1
        self._open: Dict[str, Order] = {}
        self._mid = 100.0
        self._peak_equity = self.cfg.initial_cash

    def seed_background(self, mid: float, imbalance_hint: float = 0.0) -> None:
        """Rebuild deeper passive depth; optional size skew for directional sims.

        ``imbalance_hint`` in roughly [-1, 1]: positive → more bid size (buy pressure).
        """
        self._mid = mid
        tick = self.cfg.tick
        base = self.cfg.background_size
        hint = float(np.clip(imbalance_hint, -0.95, 0.95))
        bid_mult = 1.0 + hint
        ask_mult = 1.0 - hint
        self.book = OrderBook(tick=tick)
        self._open.clear()
        for i in range(3, self.cfg.depth_levels + 3):
            bid_px = self.book.round_price(mid - i * tick)
            ask_px = self.book.round_price(mid + i * tick)
            depth_scale = 1.0 / i
            self._place_background(Side.BUY, bid_px, base * bid_mult * depth_scale)
            self._place_background(Side.SELL, ask_px, base * ask_mult * depth_scale)

    def apply_requests(self, requests: List[OrderRequest], ts: float) -> List[Fill]:
        out: List[Fill] = []
        for req in requests:
            if req.action is OrderAction.CANCEL:
                if req.order_id and self.book.cancel(req.order_id):
                    self._open.pop(req.order_id, None)
                continue
            if req.action is OrderAction.PLACE:
                if req.side is None or req.size is None or req.size <= 0:
                    continue
                price = req.price if req.price is not None else self._mid
                oid = self._new_id()
                order = Order(
                    order_id=oid,
                    side=req.side,
                    price=float(price),
                    size=float(req.size),
                    remaining=float(req.size),
                    order_type=req.order_type or OrderType.LIMIT,
                    client_tag=req.client_tag,
                    ts=ts,
                )
                fills = self.book.add(order)
                if order.remaining > 1e-12:
                    self._open[oid] = order
                for fill in fills:
                    strat_fill = Fill(
                        order_id=order.order_id,
                        side=order.side,
                        price=fill.price,
                        size=fill.size,
                        ts=fill.ts,
                        client_tag=order.client_tag,
                    )
                    self._apply_fill(strat_fill, is_strategy=True)
                    out.append(strat_fill)
        return out

    def cancel_all_strategy(self) -> None:
        for oid in list(self._open):
            self.book.cancel(oid)
            self._open.pop(oid, None)

    def open_order_ids(self) -> List[str]:
        return list(self._open)

    def maybe_external_taker(self, ts: float) -> List[Fill]:
        """Poisson-like aggressive flow; nearer-touch quotes more likely hit."""
        top = self.book.top(ts=ts)
        if top is None:
            return []
        if self.rng.random() > self.cfg.taker_intensity:
            return []
        buy_edge = max(0.0, top.ask - self._mid)
        sell_edge = max(0.0, self._mid - top.bid)
        w_buy = 1.0 / (self.cfg.tick + buy_edge)
        w_sell = 1.0 / (self.cfg.tick + sell_edge)
        side = Side.BUY if self.rng.random() < w_buy / (w_buy + w_sell) else Side.SELL
        size = max(self.cfg.tick, float(self.rng.exponential(self.cfg.taker_size_mean)))
        raw_fills = self.book.aggressive_hit(side, size, ts)
        strategy_fills: List[Fill] = []
        for fill in raw_fills:
            is_ours = fill.order_id in self._open or (
                fill.client_tag is not None
                and (
                    fill.client_tag.startswith("mm_")
                    or fill.client_tag.startswith("dir_")
                    or fill.client_tag.startswith("hybrid_")
                )
            )
            if is_ours:
                if fill.order_id in self._open:
                    o = self._open[fill.order_id]
                    if o.remaining <= 1e-12:
                        self._open.pop(fill.order_id, None)
                self._apply_fill(fill, is_strategy=True)
                strategy_fills.append(fill)
                if fill.side is Side.BUY:
                    self.metrics.spread_captured.append(self._mid - fill.price)
                else:
                    self.metrics.spread_captured.append(fill.price - self._mid)
            else:
                self.metrics.fills += 1
        return strategy_fills

    def mark(self, mid: float) -> float:
        self._mid = mid
        eq = self.pos.equity(mid)
        self.metrics.inventory_path.append(self.pos.qty)
        self.metrics.equity_path.append(eq)
        self._peak_equity = max(self._peak_equity, eq)
        self.metrics.max_drawdown = max(
            self.metrics.max_drawdown, self._peak_equity - eq
        )
        return eq

    def _place_background(self, side: Side, price: float, size: float) -> None:
        if size <= 0:
            return
        oid = self._new_id("bg")
        order = Order(
            order_id=oid,
            side=side,
            price=price,
            size=size,
            remaining=size,
            client_tag=None,
        )
        self.book.add(order)

    def _new_id(self, prefix: str = "o") -> str:
        oid = f"{prefix}{self._next_id}"
        self._next_id += 1
        return oid

    def _apply_fill(self, fill: Fill, *, is_strategy: bool) -> None:
        self.metrics.fills += 1
        if not is_strategy:
            return
        self.metrics.strategy_fills += 1
        qty_delta = fill.size if fill.side is Side.BUY else -fill.size
        apply_trade(self.pos, qty_delta, fill.price)
        if fill.order_id in self._open:
            o = self._open[fill.order_id]
            if o.remaining <= 1e-12:
                self._open.pop(fill.order_id, None)
