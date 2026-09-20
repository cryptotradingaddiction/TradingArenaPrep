"""Synthetic mid-price feed with noise, jumps, and peekable next shock."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FeedConfig:
    start_mid: float = 100.0
    tick: float = 0.01
    sigma: float = 0.02          # per-step diffusion of mid
    jump_prob: float = 0.002
    jump_sigma: float = 0.15
    seed: int = 0


class SyntheticFeed:
    """Random mid walk; exposes the *next* shock before it is applied.

    Runner flow for directional edge:
      1. peek_shock() → seed book imbalance correlated with upcoming move
      2. strategy acts on the book
      3. step() applies the shock to mid
    """

    def __init__(self, config: FeedConfig | None = None) -> None:
        self.cfg = config or FeedConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.mid = float(self.cfg.start_mid)
        self.t = 0.0
        self._pending_shock: float | None = None
        self.last_shock: float = 0.0

    def peek_shock(self) -> float:
        """Sample (or return cached) next mid shock without applying it."""
        if self._pending_shock is None:
            shock = float(self.rng.normal(0.0, self.cfg.sigma))
            if self.rng.random() < self.cfg.jump_prob:
                shock += float(self.rng.normal(0.0, self.cfg.jump_sigma))
            self._pending_shock = shock
        return self._pending_shock

    def step(self, dt: float = 1.0) -> float:
        self.t += dt
        shock = self.peek_shock()
        self._pending_shock = None
        self.last_shock = shock
        self.mid = max(self.cfg.tick * 10, self.mid + shock)
        self.mid = round(self.mid / self.cfg.tick) * self.cfg.tick
        return float(self.mid)
