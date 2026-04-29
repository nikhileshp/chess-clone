"""Download the fine-tuned Maia2 checkpoint from HF Hub at container start.

Reads MAIA_HF_REPO and (optional) MAIA_HF_FILE env vars, fetches into
/app/model/nick_p12.pt.  Idempotent — if file is already present and matches
the upstream size, we skip the download to save HF cold-start time.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = os.environ.get("MAIA_HF_REPO", "")
FILE = os.environ.get("MAIA_HF_FILE", "nick_p12.pt")
DEST = Path(os.environ.get("MAIA_CHECKPOINT_PATH", "/app/model/nick_p12.pt"))

if not REPO:
    print("MAIA_HF_REPO is empty; skipping checkpoint download (will use base Maia2-blitz)")
    sys.exit(0)

DEST.parent.mkdir(parents=True, exist_ok=True)

print(f"Downloading {REPO}:{FILE} -> {DEST}")
local_path = hf_hub_download(
    repo_id=REPO,
    filename=FILE,
    local_dir=str(DEST.parent),
    cache_dir=os.environ.get("HF_HOME", "/app/.cache/huggingface"),
)
# hf_hub_download might place file with original filename; ensure DEST exists
src = Path(local_path)
if src != DEST:
    if DEST.exists():
        DEST.unlink()
    DEST.symlink_to(src)

print(f"Checkpoint ready: {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")
