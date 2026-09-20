"""Shared dataclasses for the fake exchange and strategy contract."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"


class OrderAction(str, Enum):
    PLACE = "place"
    CANCEL = "cancel"


@dataclass(frozen=True)
class Level:
    price: float
    size: float


@dataclass(frozen=True)
class BookTop:
    """Top-of-book snapshot (+ optional depth)."""

    bid: float
    ask: float
    bid_size: float
    ask_size: float
    mid: float
    bids: tuple[Level, ...] = ()
    asks: tuple[Level, ...] = ()
    ts: float = 0.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class OrderRequest:
    """Strategy → exchange intent."""

    action: OrderAction
    side: Optional[Side] = None
    price: Optional[float] = None
    size: Optional[float] = None
    order_type: OrderType = OrderType.LIMIT
    order_id: Optional[str] = None  # required for CANCEL
    client_tag: Optional[str] = None


@dataclass
class Order:
    order_id: str
    side: Side
    price: float
    size: float
    remaining: float
    order_type: OrderType = OrderType.LIMIT
    client_tag: Optional[str] = None
    ts: float = 0.0


@dataclass(frozen=True)
class Fill:
    order_id: str
    side: Side
    price: float
    size: float
    ts: float
    client_tag: Optional[str] = None


@dataclass
class Position:
    qty: float = 0.0
    cash: float = 0.0
    avg_entry: float = 0.0
    realized_pnl: float = 0.0

    def equity(self, mid: float) -> float:
        return self.cash + self.qty * mid
