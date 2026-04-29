# chess-clone

Personal chess style cloning pipeline for **nick_p12** (Lichess + Chess.com).

Goal: feed your own ~30k blitz/rapid games into Maia-Individual to produce an
engine that plays your openings and roughly your strength (~2100 blitz peak).
This repo handles the **data side**: download, filter, dedupe, and an opening
book. Maia training itself runs in a sibling repo (see "Hand-off" below).

## Pipeline

```
                      ┌──────────────────┐
   Lichess API  ────► │ download_lichess │ ──► data/raw/lichess_blitz.pgn
                      └──────────────────┘
                      ┌──────────────────┐
   Chess.com API ───► │ download_chesscom│ ──► data/raw/chesscom_blitz_rapid.pgn
                      └──────────────────┘
                                │
                                ▼
                       ┌────────────────┐
                       │ merge_filter   │ drop bullet, <20 plies, elo<1850, dups
                       └────────────────┘
                                │
                                ▼
                       data/clean/{all,white,black}.pgn
                                │
                  ┌─────────────┼─────────────────┐
                  ▼                               ▼
           ┌────────────┐                ┌─────────────────┐
           │   stats    │                │   build_book    │
           └────────────┘                └─────────────────┘
                                                  │
                                                  ▼
                                     data/books/{white,black}.bin
                                                  │
                                                  ▼
                          → use as Polyglot book in lc0/Stockfish
                          → fine-tune Maia-1900 on data/clean/all.pgn
```

## Quick start

```bash
# 0) Sync deps (Python 3.11/3.12, uv-managed)
uv sync

# 1) Download (~5–15 minutes; chess.com is the slow side, 172 monthly archives)
uv run python scripts/download_lichess.py
uv run python scripts/download_chesscom.py

# 2) Filter + dedupe (a few minutes)
uv run python scripts/merge_filter.py

# 3) Sanity check
uv run python scripts/stats.py --pgn data/clean/all.pgn

# 4) Opening books (one for each color)
uv run python scripts/build_book.py --pgn data/clean/white.pgn --out data/books/white.bin
uv run python scripts/build_book.py --pgn data/clean/black.pgn --out data/books/black.bin
```

## Filter rules (set in `merge_filter.py`)

| Rule | Default | Why |
|---|---|---|
| Drop bullet | base time < 180s | Bullet moves are tactical noise, not style |
| Drop short games | < 20 plies | Resigns and abandons |
| Drop low-rated | user Elo < 1800 | Anchor on near-peak strength, not 2014-era you |
| Dedupe | (date, opponent, first 12 plies) | Catches re-imports across sites |

To loosen / tighten: `uv run python scripts/merge_filter.py --min-elo 1900 --min-plies 30
# or to relax: uv run python scripts/merge_filter.py --min-elo 1700`.

## Hand-off to Maia training

Maia-Individual lives in a separate repo with its own (older) deps. After
running this pipeline:

```bash
# Sibling directory:
cd ~/Projects
git clone https://github.com/CSSLab/maia-individual
cd maia-individual
# Follow their README. Point their data prep at:
#   ~/Projects/chess_clone/data/clean/all.pgn
# Use base model: maia-1900 (closest to your blitz peak of 2181).
```

Hardware: Maia fine-tuning on ~15–20k games wants a single modern NVIDIA GPU
(RTX 30/40 class, 8+ GB VRAM) and ~2–6 hours. **AMD/CPU-only is not viable**
for the training step — see [CLOUD_TRAIN.md](CLOUD_TRAIN.md) for the cloud
GPU workflow. Inference (playing against the trained model) runs fine on CPU.

## Outputs

- `data/raw/` — raw API dumps, regenerable; gitignored.
- `data/clean/` — filtered training set; gitignored.
- `data/books/` — Polyglot opening books; gitignored.
- `data/clean/dropped.txt` — drop-reason summary (sanity check).
