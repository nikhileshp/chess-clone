"""Compress data/clean/all.pgn into data/clean/all.pgn.zst.

Maia2 ingests `.pgn.zst` directly. zstd gives ~3-4x better ratio than gzip
on PGN text and decompresses fast enough that we can stream-train from it.
This is also what fits the file under GitHub's 50 MB warning threshold.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def compress(src: Path, dst: Path, level: int = 19) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source PGN missing: {src}")
    if shutil.which("zstd") is None:
        raise RuntimeError(
            "`zstd` CLI not found. Install with `sudo apt install zstd` or "
            "`brew install zstd`."
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["zstd", f"-{level}", "--force", "-o", str(dst), str(src)]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    src_size = src.stat().st_size
    dst_size = dst.stat().st_size
    logger.info(
        "Compressed %.1f MB -> %.1f MB (ratio %.2fx) at %s",
        src_size / 1e6,
        dst_size / 1e6,
        src_size / max(dst_size, 1),
        dst,
    )
    if dst_size > 95 * 1024 * 1024:
        logger.warning(
            "Result is %.1f MB — GitHub blocks files > 100 MB. Consider "
            "raising --min-elo in merge_filter.py or using Git LFS.",
            dst_size / 1e6,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path("data/clean/all.pgn"))
    parser.add_argument("--dst", type=Path, default=Path("data/clean/all.pgn.zst"))
    parser.add_argument("--level", type=int, default=19, help="zstd level 1..22 (higher = smaller)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    compress(args.src, args.dst, level=args.level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
