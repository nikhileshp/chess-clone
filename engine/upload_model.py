"""Push a fine-tuned checkpoint to a HF model repo.

Run locally after downloading the .pt from Colab.

Usage:
    python engine/upload_model.py /path/to/maia2_finetuned_best.pt \\
        --repo nikhileshp12/nick-p12-bot

Requires `huggingface_hub` and a token with write scope (one-time setup):
    uv add huggingface_hub
    uv run hf auth login   # paste an access token from huggingface.co/settings/tokens
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, help="Path to the .pt file")
    parser.add_argument(
        "--repo",
        default="nikhileshp12/nick-p12-bot",
        help="HF model repo (will be created if missing)",
    )
    parser.add_argument(
        "--filename", default="nick_p12.pt", help="Path under the HF repo to upload to"
    )
    parser.add_argument("--private", action="store_true", help="Create the repo as private")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    api = HfApi()
    print(f"Ensuring repo exists: {args.repo} (private={args.private})")
    api.create_repo(repo_id=args.repo, repo_type="model", exist_ok=True, private=args.private)

    size_mb = args.checkpoint.stat().st_size / 1e6
    print(f"Uploading {args.checkpoint} ({size_mb:.1f} MB) -> {args.repo}:{args.filename}")
    api.upload_file(
        path_or_fileobj=str(args.checkpoint),
        path_in_repo=args.filename,
        repo_id=args.repo,
        repo_type="model",
        commit_message=f"Upload {args.checkpoint.name} ({size_mb:.1f} MB)",
    )
    print(f"Done. Visible at https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
