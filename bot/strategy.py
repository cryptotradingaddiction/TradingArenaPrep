"""
Submission-shaped strategy entry point.

Day-to-day this dispatches to MM, directional, or hybrid helpers. Before contest,
flatten the chosen helper into this file if the judge requires a single module.
"""

from __future__ import annotations

from typing import List, Protocol

from bot.directional_imbalance import DirectionalImbalanceStrategy
from bot.hybrid_mm_dir import HybridMMDirStrategy
from bot.mm_inventory import MMInventoryStrategy
from bot.types import BookTop, Fill, OrderRequest


class _Inner(Protocol):
    def on_start(self, config: dict) -> None: ...
    def on_book(self, top: BookTop, ts: float) -> List[OrderRequest]: ...
    def on_fill(self, fill: Fill) -> None: ...
    def on_end(self) -> None: ...
    def register_open(self, order_ids: list[str]) -> None: ...
    def sync_state(self, qty: float, cash: float) -> None: ...


class Strategy:
    """Stable callback surface for sim runner and (later) official Arena API."""

    def __init__(self, mode: str = "mm") -> None:
        mode = mode.lower().strip()
        if mode == "mm":
            self._inner: _Inner = MMInventoryStrategy()
        elif mode in {"dir", "directional", "imbalance"}:
            self._inner = DirectionalImbalanceStrategy()
        elif mode == "hybrid":
            self._inner = HybridMMDirStrategy()
        else:
            raise ValueError(
                f"Unsupported strategy mode: {mode!r} "
                f"(use 'mm', 'directional', or 'hybrid')"
            )
        self.mode = mode

    def on_start(self, config: dict) -> None:
        self._inner.on_start(config)

    def on_book(self, top: BookTop, ts: float) -> List[OrderRequest]:
        return self._inner.on_book(top, ts)

    def on_fill(self, fill: Fill) -> None:
        self._inner.on_fill(fill)

    def on_end(self) -> None:
        self._inner.on_end()

    def register_open(self, order_ids: list[str]) -> None:
        self._inner.register_open(order_ids)

    def sync_state(self, qty: float, cash: float) -> None:
        self._inner.sync_state(qty, cash)


def build_strategy(name: str = "mm") -> Strategy:
    return Strategy(mode=name)
