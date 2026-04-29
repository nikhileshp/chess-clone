"""Sanity-check stats for a cleaned PGN file.

Reports:
  * total game count
  * year histogram
  * time-control bucket histogram
  * user rating distribution (min/p10/median/p90/max)
  * color split
  * top 10 first-move sequences (length 4) for white and black
"""
from __future__ import annotations

import argparse
import logging
import statistics
import sys
from collections import Counter
from pathlib import Path

import chess.pgn
from tqdm import tqdm

logger = logging.getLogger(__name__)


def bucket_tc(tc: str) -> str:
    if not tc or tc == "-":
        return "unknown"
    if "/" in tc:
        return "correspondence"
    base = tc.split("+", 1)[0]
    try:
        b = int(base)
    except ValueError:
        return "unknown"
    if b < 180:
        return "bullet"
    if b < 600:
        return "blitz"
    if b < 1800:
        return "rapid"
    return "classical"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", type=Path, default=Path("data/clean/all.pgn"))
    parser.add_argument("--user", default="nick_p12")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.pgn.exists():
        logger.error("Missing: %s", args.pgn)
        return 1

    user = args.user.lower()
    years: Counter = Counter()
    tcs: Counter = Counter()
    user_elos: list[int] = []
    color_split: Counter = Counter()
    white_openings: Counter = Counter()
    black_openings: Counter = Counter()
    total = 0

    with args.pgn.open("r", encoding="utf-8", errors="replace") as f:
        pbar = tqdm(desc=args.pgn.name)
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            total += 1
            pbar.update(1)
            h = game.headers
            white = (h.get("White") or "").lower()
            user_is_white = white == user
            color_split["white" if user_is_white else "black"] += 1

            date = h.get("UTCDate") or h.get("Date") or ""
            year = date[:4] if date and date[:4].isdigit() else "????"
            years[year] += 1
            tcs[bucket_tc(h.get("TimeControl") or "")] += 1

            elo_key = "WhiteElo" if user_is_white else "BlackElo"
            try:
                user_elos.append(int(h.get(elo_key, "")))
            except ValueError:
                pass

            moves = []
            node = game
            while node.variations and len(moves) < 4:
                node = node.variation(0)
                moves.append(node.san())
            if len(moves) == 4:
                key = " ".join(moves)
                (white_openings if user_is_white else black_openings)[key] += 1
        pbar.close()

    print(f"\n=== {args.pgn} ===")
    print(f"Total games: {total}")
    print(f"\nColor split: {dict(color_split)}")
    print("\nTime-control buckets:")
    for k, v in tcs.most_common():
        print(f"  {k:14s} {v}")
    print("\nYear histogram:")
    for y in sorted(years):
        bar = "#" * min(60, years[y] // 50 or 1)
        print(f"  {y}  {years[y]:6d}  {bar}")
    if user_elos:
        user_elos.sort()
        n = len(user_elos)
        p10 = user_elos[n // 10]
        p90 = user_elos[(9 * n) // 10]
        print("\nUser Elo distribution:")
        print(
            f"  min={user_elos[0]}  p10={p10}  median={int(statistics.median(user_elos))} "
            f"p90={p90}  max={user_elos[-1]}  n={n}"
        )
    print("\nTop 10 White openings (4-ply):")
    for k, v in white_openings.most_common(10):
        print(f"  {v:5d}  {k}")
    print("\nTop 10 Black openings (4-ply):")
    for k, v in black_openings.most_common(10):
        print(f"  {v:5d}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
