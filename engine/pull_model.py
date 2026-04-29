"""Download the fine-tuned Maia2 checkpoint + time-mimic LightGBM models from
HF Hub at container start.

Reads MAIA_HF_REPO and (optional) MAIA_HF_FILE env vars, fetches the Maia
checkpoint into /app/model/nick_p12.pt, and the time-mimic models into
/app/weights/time_model/{time_q25.lgb, time_q50.lgb, time_q75.lgb, meta.json}.

Idempotent (HF cache handles re-runs cheaply). Time-mimic download is
best-effort: if files are missing on the HF repo, the engine falls back to
fixed-depth Stockfish ranking.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = os.environ.get("MAIA_HF_REPO", "")
FILE = os.environ.get("MAIA_HF_FILE", "nick_p12.pt")
DEST = Path(os.environ.get("MAIA_CHECKPOINT_PATH", "/app/model/nick_p12.pt"))
TIME_MODEL_DIR = Path(os.environ.get("MAIA_TIME_MODEL_DIR", "/app/weights/time_model"))
TIME_MODEL_FILES = ("time_q25.lgb", "time_q50.lgb", "time_q75.lgb", "meta.json")

if not REPO:
    print("MAIA_HF_REPO is empty; skipping all downloads (will use base Maia2, no time mimicry)")
    sys.exit(0)

DEST.parent.mkdir(parents=True, exist_ok=True)
TIME_MODEL_DIR.mkdir(parents=True, exist_ok=True)
cache_dir = os.environ.get("HF_HOME", "/app/.cache/huggingface")

# --- Maia checkpoint ---
print(f"Downloading {REPO}:{FILE} -> {DEST}")
local_path = hf_hub_download(
    repo_id=REPO,
    filename=FILE,
    local_dir=str(DEST.parent),
    cache_dir=cache_dir,
)
src = Path(local_path)
if src != DEST:
    if DEST.exists():
        DEST.unlink()
    DEST.symlink_to(src)
print(f"Checkpoint ready: {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")

# --- Time-mimic LightGBM models (best-effort) ---
print(f"Downloading time-mimic models from {REPO}:time_model/* -> {TIME_MODEL_DIR}")
all_present = True
for fname in TIME_MODEL_FILES:
    target = TIME_MODEL_DIR / fname
    try:
        local = hf_hub_download(
            repo_id=REPO,
            filename=f"time_model/{fname}",
            local_dir=str(TIME_MODEL_DIR.parent),
            cache_dir=cache_dir,
        )
        local_p = Path(local)
        if local_p != target:
            if target.exists():
                target.unlink()
            target.symlink_to(local_p)
        print(f"  {fname}: {target.stat().st_size / 1024:.1f} KB")
    except Exception as e:  # noqa: BLE001
        print(f"  {fname}: FAILED ({e})")
        all_present = False

if all_present:
    print(f"Time-mimic ready: {TIME_MODEL_DIR}")
else:
    print("Time-mimic models incomplete; engine will fall back to fixed-depth ranking.")
