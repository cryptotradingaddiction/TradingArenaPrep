# ArenaPrep — AlgoTrade Arena LOB lab

Local practice lab for **Qminers AlgoTrade Arena**.

This folder is independent of the DCh thesis repo. No TISEAN / PyRQA.

## Setup

```bat
cd /d C:\ArenaPrep
py -3 -m pip install -r requirements.txt
```

## Run simulations

Market making (spread + inventory skew):

```bat
py -3 -m sim.runner --strategy mm --steps 1000 --seed 1
```

Directional (imbalance → mid lean):

```bat
py -3 -m sim.runner --strategy directional --steps 1000 --seed 1
```

## Tests

```bat
py -3 -m unittest discover -s tests -v
```

## Layout

- `bot/strategy.py` — submission-shaped entry (`Strategy` callbacks)
- `bot/mm_inventory.py` — Avellaneda–Stoikov-style MM with inventory skew
- `bot/directional_imbalance.py` — imbalance signal → one-sided / taker lean
- `sim/` — fake price-time LOB, exchange, synthetic feed, runner

## Strategy contract

```python
class Strategy:
    def on_start(self, config: dict) -> None: ...
    def on_book(self, top: BookTop, ts: float) -> list[OrderRequest]: ...
    def on_fill(self, fill: Fill) -> None: ...
    def on_end(self) -> None: ...
```

When the official Arena API lands, add an adapter that maps their callbacks onto this interface. Before contest day, flatten helpers into a single `strategy.py` if the judge requires one file.

## Phase roadmap

1. Done: market making (spread + inventory skew)
2. Done: directional imbalance → mid move (wired in runner)
3. Later: hybrid MM + directional skew, official API adapter
