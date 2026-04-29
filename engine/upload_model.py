"""Push a fine-tuned checkpoint or a directory of artifacts to a HF model repo.

Run locally after downloading the .pt from Colab, or after training the
time-mimic LightGBM models.

Usage (single file):
    python engine/upload_model.py /path/to/maia2_finetuned_best.pt \\
        --repo nikhileshp12/nick-p12-bot

Usage (directory — uploads every file under it, preserving structure inside repo):
    python engine/upload_model.py weights/time_model \\
        --repo nikhileshp12/nick-p12-bot \\
        --path-in-repo time_model

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
    parser.add_argument("path", type=Path, help="Path to a .pt file or a directory")
    parser.add_argument(
        "--repo",
        default="nikhileshp12/nick-p12-bot",
        help="HF model repo (will be created if missing)",
    )
    parser.add_argument(
        "--filename",
        default=None,
        help="(File mode) Path under the HF repo to upload to. Defaults to the file's basename.",
    )
    parser.add_argument(
        "--path-in-repo",
        default=None,
        help="(Dir mode) Subdirectory in the HF repo to upload into. Defaults to the dir's basename.",
    )
    parser.add_argument("--private", action="store_true", help="Create the repo as private")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"Path not found: {args.path}", file=sys.stderr)
        return 1

    api = HfApi()
    print(f"Ensuring repo exists: {args.repo} (private={args.private})")
    api.create_repo(repo_id=args.repo, repo_type="model", exist_ok=True, private=args.private)

    if args.path.is_dir():
        path_in_repo = args.path_in_repo or args.path.name
        files = sorted(p for p in args.path.iterdir() if p.is_file())
        if not files:
            print(f"No files in {args.path}", file=sys.stderr)
            return 1
        print(f"Uploading {len(files)} files from {args.path} -> {args.repo}:{path_in_repo}/")
        for fp in files:
            target = f"{path_in_repo}/{fp.name}"
            size_kb = fp.stat().st_size / 1024
            print(f"  {fp.name} ({size_kb:.1f} KB) -> {target}")
            api.upload_file(
                path_or_fileobj=str(fp),
                path_in_repo=target,
                repo_id=args.repo,
                repo_type="model",
                commit_message=f"Upload {target}",
            )
    else:
        filename = args.filename or args.path.name
        size_mb = args.path.stat().st_size / 1e6
        print(f"Uploading {args.path} ({size_mb:.1f} MB) -> {args.repo}:{filename}")
        api.upload_file(
            path_or_fileobj=str(args.path),
            path_in_repo=filename,
            repo_id=args.repo,
            repo_type="model",
            commit_message=f"Upload {args.path.name} ({size_mb:.1f} MB)",
        )

    print(f"Done. Visible at https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
