# Cloud training notes

Local hardware is **AMD with 128 GB unified memory, no NVIDIA GPU**.
That is fine for everything in this repo (data prep, opening books) and for
**inference** (running the trained Maia network via lc0 on CPU). It is
**not viable for fine-tuning** Maia / Leela: the training code is CUDA-only,
and unified memory is irrelevant when the bottleneck is GPU compute.

## Recommended cloud path

Use a single GPU instance on Vast.ai or RunPod for the training step only.
Keep all data prep local — the cloud box just receives a cleaned PGN and
returns weights.

| Provider | GPU | Approx $/hr | Notes |
|---|---|---|---|
| Vast.ai | RTX 4090 (24 GB) | $0.30–0.50 | Cheapest, spot-style |
| RunPod | RTX A6000 (48 GB) | $0.79 | More stable, official |
| Lambda | A100 (40/80 GB) | $1.10–2.00 | Overkill for ~15k games |

Estimated end-to-end cost: **$2–10** for one Maia-Individual fine-tune.

## Workflow

```bash
# 1) Local: prep data
uv run python scripts/download_lichess.py
uv run python scripts/download_chesscom.py
uv run python scripts/merge_filter.py
uv run python scripts/stats.py  # sanity check

# 2) Local: ship clean PGN to cloud box (~50–200 MB after compression)
gzip -k data/clean/all.pgn
scp data/clean/all.pgn.gz user@cloud-box:~/maia/

# 3) Cloud box (Ubuntu 22.04 + CUDA 11.8 + Python 3.7 — Maia is older code):
git clone https://github.com/CSSLab/maia-individual ~/maia/repo
cd ~/maia/repo
# Follow their README. Two key entry points:
#   move_prediction/maia_chess_backend/data_processing/  (PGN → training tensors)
#   move_prediction/main_train.py                        (fine-tune from base)
# Use base: maia-1900 (closest to your blitz peak of 2181).

# 4) Cloud box: pull weights back to local
# Outputs: a Leela-format weights file (gzipped protobuf, ~10–50 MB)
scp user@cloud-box:~/maia/output/final_weights.pb.gz ./weights/

# 5) Local: play against it
# lc0 runs on CPU just fine; pair with your Polyglot books.
lc0 --weights=weights/final_weights.pb.gz --backend=blas
# Then load in any UCI-compatible GUI (Cute Chess, Arena, BanksiaGUI).
```

## Why not train on CPU locally?

Leela's training step is matrix-mul-bound on a policy/value net with thousands
of small batches per epoch. A single epoch on 15k games:
- 4090: ~10 minutes
- 7950X CPU: ~6–10 hours

Maia-Individual papers show useful convergence around 5–20 epochs. Worst
case on CPU is days; cloud GPU is hours.

## When unified memory IS useful

Inference, not training. Once you have the trained `.pb.gz`:
- `lc0 --backend=blas` runs your network with no GPU, leveraging your 128 GB.
- You can spin up multiple lc0 instances at different node-counts to spar
  against versions of yourself at different "thinking depths."
- Self-play data generation (if you ever want to extend the model) also runs
  on CPU — slow per game but parallelizable across your cores.
