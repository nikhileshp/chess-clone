"""Stream blitz games for a Lichess user into a single PGN file.

Lichess Open API: https://lichess.org/api#tag/Games/operation/apiGamesUser
Endpoint streams NDJSON-or-PGN at ~30 games/sec for an authenticated session.
We request PGN directly with `?perfType=blitz` to skip bullet/rapid/variants.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

LICHESS_GAMES_URL = "https://lichess.org/api/games/user/{user}"


def download(user: str, out_path: Path, perf_type: str = "blitz", token: str | None = None) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Accept": "application/x-chess-pgn"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    params = {
        "perfType": perf_type,
        "rated": "true",
        "clocks": "true",
        "evals": "true",
        "opening": "false",
        "literate": "false",
    }
    url = LICHESS_GAMES_URL.format(user=user)

    logger.info("Streaming %s games for user=%s to %s", perf_type, user, out_path)
    started = time.time()
    bytes_written = 0
    games_seen = 0

    with requests.get(url, params=params, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                bytes_written += len(chunk)
                # Rough progress: count "[Event " occurrences in chunk
                games_seen += chunk.count(b"[Event ")
                if games_seen and games_seen % 500 == 0:
                    logger.info("  ~%d games | %.1f MB", games_seen, bytes_written / 1e6)

    elapsed = time.time() - started
    logger.info(
        "Done: ~%d games | %.1f MB | %.0fs | %s",
        games_seen,
        bytes_written / 1e6,
        elapsed,
        out_path,
    )
    return games_seen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="nick_p12")
    parser.add_argument("--perf", default="blitz", choices=["blitz", "rapid", "classical"])
    parser.add_argument("--out", type=Path, default=Path("data/raw/lichess_blitz.pgn"))
    parser.add_argument("--token", default=None, help="Optional Lichess API token (raises rate cap)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    download(args.user, args.out, perf_type=args.perf, token=args.token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
