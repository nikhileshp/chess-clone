"""Merge raw lichess + chess.com PGNs into a clean training set.

Filters applied:
  * Drop games where time control is bullet (any chess.com leftovers, or
    lichess <180s base + 0 increment if perfType slipped).
  * Drop games shorter than MIN_PLIES (default 20 = 10 full moves).
  * Drop games where the user's rating-at-time is below MIN_USER_ELO.
  * Dedupe by (date, opponent, first 12 plies).

Outputs four PGN files:
  data/clean/all.pgn       - all kept games
  data/clean/white.pgn     - games where the user played white
  data/clean/black.pgn     - games where the user played black
  data/clean/dropped.txt   - one-line reason per dropped game (sample)
"""
from __future__ import annotations

import argparse
import io
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import chess.pgn
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FilterConfig:
    user: str
    min_plies: int = 20
    min_user_elo: int = 1800
    bullet_max_base_seconds: int = 179  # < 3 minutes base = bullet


def is_user_white(headers: dict, user: str) -> bool | None:
    white = (headers.get("White") or "").lower()
    black = (headers.get("Black") or "").lower()
    u = user.lower()
    if white == u:
        return True
    if black == u:
        return False
    return None


def parse_user_elo(headers: dict, user_is_white: bool) -> int | None:
    key = "WhiteElo" if user_is_white else "BlackElo"
    raw = headers.get(key, "").strip()
    if not raw or raw == "?":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def is_bullet(headers: dict, cfg: FilterConfig) -> bool:
    tc = (headers.get("TimeControl") or "").strip()
    if not tc or tc == "-":
        return False
    # Forms: "60", "60+0", "180+2", "1/259200" (correspondence)
    if "/" in tc:  # correspondence
        return False
    base_part = tc.split("+", 1)[0]
    try:
        base = int(base_part)
    except ValueError:
        return False
    return base <= cfg.bullet_max_base_seconds


def first_n_plies(game: chess.pgn.Game, n: int) -> tuple[str, ...]:
    moves: list[str] = []
    node = game
    while node.variations and len(moves) < n:
        node = node.variation(0)
        moves.append(node.san())
    return tuple(moves)


def count_plies(game: chess.pgn.Game) -> int:
    n = 0
    node = game
    while node.variations:
        node = node.variation(0)
        n += 1
    return n


def iter_games(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                return
            yield game


def serialize(game: chess.pgn.Game) -> str:
    buf = io.StringIO()
    exporter = chess.pgn.FileExporter(buf)
    game.accept(exporter)
    return buf.getvalue()


def process(inputs: list[Path], out_dir: Path, cfg: FilterConfig) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / "all.pgn"
    white_path = out_dir / "white.pgn"
    black_path = out_dir / "black.pgn"
    dropped_path = out_dir / "dropped.txt"

    seen: set[tuple] = set()
    drop_reasons: Counter = Counter()

    fall = all_path.open("w", encoding="utf-8")
    fwhite = white_path.open("w", encoding="utf-8")
    fblack = black_path.open("w", encoding="utf-8")
    fdrop = dropped_path.open("w", encoding="utf-8")
    kept = 0
    total = 0

    try:
        for src in inputs:
            if not src.exists():
                logger.warning("Skipping missing input: %s", src)
                continue
            logger.info("Reading %s", src)
            for game in tqdm(iter_games(src), desc=src.name):
                total += 1
                h = dict(game.headers)
                user_is_white = is_user_white(h, cfg.user)
                if user_is_white is None:
                    drop_reasons["not_user"] += 1
                    continue
                if is_bullet(h, cfg):
                    drop_reasons["bullet"] += 1
                    continue
                user_elo = parse_user_elo(h, user_is_white)
                if user_elo is None:
                    drop_reasons["no_elo"] += 1
                    continue
                if user_elo < cfg.min_user_elo:
                    drop_reasons["low_elo"] += 1
                    continue
                plies = count_plies(game)
                if plies < cfg.min_plies:
                    drop_reasons["short_game"] += 1
                    continue
                opening_key = first_n_plies(game, 12)
                date = h.get("UTCDate") or h.get("Date") or ""
                opp = (h.get("Black") if user_is_white else h.get("White")) or ""
                dedup_key = (date, opp.lower(), opening_key)
                if dedup_key in seen:
                    drop_reasons["dup"] += 1
                    continue
                seen.add(dedup_key)
                pgn_text = serialize(game)
                fall.write(pgn_text)
                (fwhite if user_is_white else fblack).write(pgn_text)
                kept += 1
        # Drop reason summary
        fdrop.write(f"Total scanned: {total}\nKept: {kept}\nDropped reasons:\n")
        for reason, n in drop_reasons.most_common():
            fdrop.write(f"  {reason}: {n}\n")
    finally:
        fall.close()
        fwhite.close()
        fblack.close()
        fdrop.close()

    logger.info("Total scanned: %d", total)
    logger.info("Kept: %d", kept)
    for reason, n in drop_reasons.most_common():
        logger.info("  drop[%s]: %d", reason, n)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="nick_p12")
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=[
            Path("data/raw/lichess_blitz.pgn"),
            Path("data/raw/chesscom_blitz_rapid.pgn"),
        ],
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/clean"))
    parser.add_argument("--min-elo", type=int, default=1800)
    parser.add_argument("--min-plies", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = FilterConfig(user=args.user, min_user_elo=args.min_elo, min_plies=args.min_plies)
    process(args.inputs, args.out_dir, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
