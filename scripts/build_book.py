"""Build a Polyglot opening book from a cleaned PGN file.

Counts moves per Zobrist position (using python-chess polyglot zobrist hashing),
weighted by user rating (better games count more) and game result (wins > draws > losses).

Outputs:
  data/books/<name>.bin

Usage:
  python scripts/build_book.py --pgn data/clean/white.pgn --out data/books/white.bin
  python scripts/build_book.py --pgn data/clean/black.pgn --out data/books/black.bin
"""
from __future__ import annotations

import argparse
import logging
import struct
import sys
from collections import defaultdict
from pathlib import Path

import chess
import chess.pgn
import chess.polyglot
from tqdm import tqdm

logger = logging.getLogger(__name__)

MAX_PLIES_DEFAULT = 16  # 8 full moves


def result_weight(result: str, user_is_white: bool) -> float:
    if result == "1-0":
        return 2.0 if user_is_white else 0.5
    if result == "0-1":
        return 0.5 if user_is_white else 2.0
    if result == "1/2-1/2":
        return 1.0
    return 0.5


def encode_move(move: chess.Move) -> int:
    """Polyglot move encoding: bits 0-5 to_sq, 6-11 from_sq, 12-14 promotion."""
    to_sq = move.to_square
    from_sq = move.from_square
    promo = 0
    if move.promotion:
        promo = {chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}.get(
            move.promotion, 0
        )
    return (promo << 12) | (from_sq << 6) | to_sq


def build(pgn_path: Path, out_path: Path, user: str, max_plies: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    user_lower = user.lower()
    # key -> {move -> weight}
    table: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))

    games_seen = 0
    with pgn_path.open("r", encoding="utf-8", errors="replace") as f:
        pbar = tqdm(desc=pgn_path.name)
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games_seen += 1
            pbar.update(1)

            h = game.headers
            user_is_white = (h.get("White") or "").lower() == user_lower
            result = h.get("Result", "*")
            w_result = result_weight(result, user_is_white)

            board = game.board()
            node = game
            ply = 0
            while node.variations and ply < max_plies:
                node = node.variation(0)
                move = node.move
                # Only record the user's own moves (those they actually chose)
                user_to_move = (board.turn == chess.WHITE) == user_is_white
                if user_to_move:
                    key = chess.polyglot.zobrist_hash(board)
                    table[key][encode_move(move)] += w_result
                board.push(move)
                ply += 1
        pbar.close()

    # Polyglot file: sorted by key, each entry is 16 bytes:
    #   uint64 key (BE), uint16 move (BE), uint16 weight (BE), uint32 learn (BE)
    entries: list[tuple[int, int, int]] = []
    for key, moves in table.items():
        # Normalize so the most-frequent move gets weight ~65535
        max_w = max(moves.values())
        if max_w <= 0:
            continue
        for mv, w in moves.items():
            scaled = max(1, min(65535, int(round(w / max_w * 65535))))
            entries.append((key, mv, scaled))
    entries.sort(key=lambda x: (x[0], -x[2]))

    with out_path.open("wb") as out:
        for key, mv, w in entries:
            out.write(struct.pack(">QHHI", key, mv, w, 0))

    logger.info(
        "Book written: %s | games=%d | positions=%d | entries=%d",
        out_path,
        games_seen,
        len(table),
        len(entries),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--user", default="nick_p12")
    parser.add_argument("--max-plies", type=int, default=MAX_PLIES_DEFAULT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build(args.pgn, args.out, args.user, args.max_plies)
    return 0


if __name__ == "__main__":
    sys.exit(main())
