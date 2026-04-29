"""Walk Chess.com monthly archives for a user and write blitz+rapid games.

API docs: https://www.chess.com/news/view/published-data-api
Each archive URL returns JSON with a `games` list; each game has a `pgn` string,
`time_class` ("blitz"/"rapid"/"bullet"/"daily"), and `rules` ("chess"/"chess960"/...).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

CHESSCOM_ARCHIVES = "https://api.chess.com/pub/player/{user}/games/archives"
USER_AGENT = "chess-clone-pipeline (contact: prabhakar.nikhilesh@gmail.com)"

KEEP_TIME_CLASSES = {"blitz", "rapid"}
KEEP_RULES = {"chess"}


def list_archives(user: str) -> list[str]:
    r = requests.get(
        CHESSCOM_ARCHIVES.format(user=user),
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["archives"]


def fetch_archive(url: str) -> list[dict]:
    for attempt in range(3):
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
            if r.status_code == 429:
                wait = 2 ** attempt * 5
                logger.warning("Rate limited at %s, sleeping %ds", url, wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json().get("games", [])
        except requests.RequestException as e:
            if attempt == 2:
                raise
            logger.warning("Retry %d for %s: %s", attempt + 1, url, e)
            time.sleep(2 ** attempt)
    return []


def download(user: str, out_path: Path) -> tuple[int, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    archives = list_archives(user)
    logger.info("Found %d monthly archives for %s", len(archives), user)

    total_games = 0
    kept_games = 0
    with out_path.open("w", encoding="utf-8") as out:
        for url in tqdm(archives, desc="archives"):
            games = fetch_archive(url)
            total_games += len(games)
            for g in games:
                if g.get("time_class") not in KEEP_TIME_CLASSES:
                    continue
                if g.get("rules") not in KEEP_RULES:
                    continue
                pgn = g.get("pgn")
                if not pgn:
                    continue
                out.write(pgn.rstrip())
                out.write("\n\n")
                kept_games += 1
            # Be nice to the API
            time.sleep(0.1)

    logger.info("Wrote %d / %d games (blitz+rapid only) to %s", kept_games, total_games, out_path)
    return kept_games, total_games


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="nick_p12")
    parser.add_argument("--out", type=Path, default=Path("data/raw/chesscom_blitz_rapid.pgn"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    download(args.user, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
