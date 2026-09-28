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

Hybrid (MM quotes + imbalance skew, no taker):

```bat
py -3 -m sim.runner --strategy hybrid --steps 1000 --seed 1
```

### Realism flags (defaults on)

```bat
py -3 -m sim.runner --strategy hybrid --steps 1000 --seed 1 ^
  --fee-bps 1 --adverse-half-ticks 1 --flatten-steps 50 --quote-every 1
```

| Flag | Default | Effect |
|------|---------|--------|
| `--fee-bps` | 1 | Fee on every strategy fill |
| `--adverse-half-ticks` | 1 | Maker fills worsen by half-tick × N |
| `--flatten-steps` | 50 | Last N steps force flat inventory |
| `--quote-every` | 1 | Only requote every N book updates |

## Multi-seed benchmark

Compare mm / directional / hybrid across seeds (realism defaults on):

```bat
py -3 -m sim.benchmark --steps 500 --seeds 0-19
```

## Tests

```bat
py -3 -m unittest discover -s tests -v
```

## Layout

- `bot/strategy.py` — submission-shaped entry (`Strategy` callbacks)
- `bot/mm_inventory.py` — Avellaneda–Stoikov-style MM with inventory skew
- `bot/directional_imbalance.py` — imbalance signal → one-sided / taker lean
- `bot/hybrid_mm_dir.py` — two-sided MM + faded imbalance skew (no taker)
- `sim/` — fake LOB, exchange (fees/adverse/flatten), feed, runner, benchmark

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
3. Done: hybrid MM + directional skew
4. Done: sim realism (fees, adverse, flatten, quote_every) + multi-seed benchmark
5. Later: official API adapter
