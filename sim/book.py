"""Price-time priority limit order book (single instrument)."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple

from bot.types import BookTop, Fill, Level, Order, Side


@dataclass
class _Resting:
    order: Order
    seq: int


class OrderBook:
    """Simple LOB: bids descending, asks ascending; FIFO within a price."""

    def __init__(self, tick: float = 0.01) -> None:
        self.tick = float(tick)
        self._bids: Dict[float, Deque[_Resting]] = defaultdict(deque)
        self._asks: Dict[float, Deque[_Resting]] = defaultdict(deque)
        self._by_id: Dict[str, Tuple[Side, float]] = {}
        self._seq = 0

    def round_price(self, price: float) -> float:
        t = self.tick
        return round(round(price / t) * t, 10)

    def best_bid(self) -> Optional[float]:
        self._purge_empty(self._bids)
        return max(self._bids) if self._bids else None

    def best_ask(self) -> Optional[float]:
        self._purge_empty(self._asks)
        return min(self._asks) if self._asks else None

    def get_order(self, order_id: str) -> Optional[Order]:
        loc = self._by_id.get(order_id)
        if loc is None:
            return None
        side, price = loc
        book = self._bids if side is Side.BUY else self._asks
        for resting in book.get(price, ()):
            if resting.order.order_id == order_id:
                return resting.order
        return None

    def cancel(self, order_id: str) -> bool:
        loc = self._by_id.pop(order_id, None)
        if loc is None:
            return False
        side, price = loc
        book = self._bids if side is Side.BUY else self._asks
        q = book.get(price)
        if not q:
            book.pop(price, None)
            return False
        for i, resting in enumerate(q):
            if resting.order.order_id == order_id:
                del q[i]
                if not q:
                    del book[price]
                return True
        return False

    def add(self, order: Order) -> List[Fill]:
        """Insert a limit order; return any immediate fills (crossing)."""
        if order.remaining <= 0:
            return []
        price = self.round_price(order.price)
        order.price = price
        fills: List[Fill] = []
        if order.side is Side.BUY:
            fills.extend(self._match_buy(order))
        else:
            fills.extend(self._match_sell(order))
        if order.remaining > 1e-12:
            self._rest(order)
        return fills

    def aggressive_hit(
        self,
        side: Side,
        size: float,
        ts: float,
        *,
        limit_price: Optional[float] = None,
    ) -> List[Fill]:
        """External taker sweeps resting liquidity (no resting leftover)."""
        remaining = float(size)
        fills: List[Fill] = []
        if remaining <= 0:
            return fills
        if side is Side.BUY:
            while remaining > 1e-12 and self._asks:
                price = min(self._asks)
                if limit_price is not None and price > limit_price + 1e-12:
                    break
                level_fills = self._take_from_level(Side.SELL, price, remaining, ts)
                if not level_fills:
                    # Empty / stale level — drop key so we cannot spin forever
                    self._asks.pop(price, None)
                    continue
                fills.extend(level_fills)
                remaining -= sum(f.size for f in level_fills)
        else:
            while remaining > 1e-12 and self._bids:
                price = max(self._bids)
                if limit_price is not None and price < limit_price - 1e-12:
                    break
                level_fills = self._take_from_level(Side.BUY, price, remaining, ts)
                if not level_fills:
                    self._bids.pop(price, None)
                    continue
                fills.extend(level_fills)
                remaining -= sum(f.size for f in level_fills)
        return fills

    def top(self, depth: int = 5, ts: float = 0.0) -> Optional[BookTop]:
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None or ba is None:
            return None
        if bb > ba + 1e-12:
            # Crossed book is invalid for strategies — refuse snapshot
            return None
        bid_size = self._level_size(self._bids[bb])
        ask_size = self._level_size(self._asks[ba])
        bids = tuple(
            Level(p, self._level_size(self._bids[p]))
            for p in sorted(self._bids, reverse=True)[:depth]
        )
        asks = tuple(
            Level(p, self._level_size(self._asks[p]))
            for p in sorted(self._asks)[:depth]
        )
        return BookTop(
            bid=bb,
            ask=ba,
            bid_size=bid_size,
            ask_size=ask_size,
            mid=0.5 * (bb + ba),
            bids=bids,
            asks=asks,
            ts=ts,
        )

    def _rest(self, order: Order) -> None:
        self._seq += 1
        book = self._bids if order.side is Side.BUY else self._asks
        book[order.price].append(_Resting(order=order, seq=self._seq))
        self._by_id[order.order_id] = (order.side, order.price)

    def _match_buy(self, order: Order) -> List[Fill]:
        fills: List[Fill] = []
        while order.remaining > 1e-12 and self._asks:
            best = min(self._asks)
            if order.price + 1e-12 < best:
                break
            level_fills = self._take_from_level(Side.SELL, best, order.remaining, order.ts)
            if not level_fills:
                self._asks.pop(best, None)
                continue
            fills.extend(level_fills)
            order.remaining -= sum(f.size for f in level_fills)
        return fills

    def _match_sell(self, order: Order) -> List[Fill]:
        fills: List[Fill] = []
        while order.remaining > 1e-12 and self._bids:
            best = max(self._bids)
            if order.price - 1e-12 > best:
                break
            level_fills = self._take_from_level(Side.BUY, best, order.remaining, order.ts)
            if not level_fills:
                self._bids.pop(best, None)
                continue
            fills.extend(level_fills)
            order.remaining -= sum(f.size for f in level_fills)
        return fills

    def _take_from_level(
        self,
        maker_side: Side,
        price: float,
        qty: float,
        ts: float,
    ) -> List[Fill]:
        book = self._bids if maker_side is Side.BUY else self._asks
        q = book.get(price)
        if not q:
            book.pop(price, None)
            return []
        fills: List[Fill] = []
        need = qty
        while need > 1e-12 and q:
            resting = q[0]
            take = min(need, resting.order.remaining)
            resting.order.remaining -= take
            need -= take
            fills.append(
                Fill(
                    order_id=resting.order.order_id,
                    side=resting.order.side,
                    price=price,
                    size=take,
                    ts=ts,
                    client_tag=resting.order.client_tag,
                )
            )
            if resting.order.remaining <= 1e-12:
                q.popleft()
                self._by_id.pop(resting.order.order_id, None)
        if not q:
            del book[price]
        return fills

    @staticmethod
    def _purge_empty(book: Dict[float, Deque[_Resting]]) -> None:
        dead = [px for px, q in book.items() if not q]
        for px in dead:
            del book[px]

    @staticmethod
    def _level_size(queue: Iterable[_Resting]) -> float:
        return float(sum(r.order.remaining for r in queue))
