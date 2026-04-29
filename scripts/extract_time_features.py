"""Extract per-move time-mimicry features from raw PGN files.

Reads chess.com and Lichess PGNs (with `%clk` annotations), and emits one CSV
row per move played by USER. Output feeds the LightGBM time-prediction model.

Run per source:
    uv run python scripts/extract_time_features.py \
        --source chesscom --pgn data/raw/chesscom_blitz_rapid.pgn \
        --min-year 2020 --out data/processed/time_features_chesscom.csv

    uv run python scripts/extract_time_features.py \
        --source lichess --pgn data/raw/lichess_blitz.pgn \
        --out data/processed/time_features_lichess_blitz.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path
from typing import Iterator, Optional

import chess
import chess.pgn

logger = logging.getLogger(__name__)

USER = "nick_p12"
MIN_TIME_SPENT = 0.05         # seconds; below this is premove/lag
SKIP_OPENING_PLIES = 4         # drop first 2 moves per side
EVAL_CLIP_CP = 1000            # cap eval magnitude
MIN_INITIAL_TIME_S = 30        # filter out absurdly short games

CLOCK_RE = re.compile(r"\[%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]")
EVAL_NUM_RE = re.compile(r"\[%eval\s+(-?\d+(?:\.\d+)?)\]")
EVAL_MATE_RE = re.compile(r"\[%eval\s+#(-?\d+)\]")

PIECE_VALUE_CP = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

CSV_FIELDS = [
    "source", "game_id", "date",
    "time_control_class", "initial_time", "increment",
    "my_color", "elo_expected_score",
    "ply", "move_number", "phase",
    "num_pieces", "material_balance_cp",
    "in_check", "num_legal_moves", "is_recapture",
    "expected_moves_left",
    "eval_cp", "abs_eval_cp", "eval_delta",
    "my_time_left", "opp_time_left", "my_time_frac", "time_diff",
    "opp_last_move_time",
    "fen",
    "time_spent",  # LABEL
]


def parse_clock(comment: str) -> Optional[float]:
    m = CLOCK_RE.search(comment)
    if not m:
        return None
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def parse_eval_cp(comment: str) -> Optional[int]:
    """Eval in centipawns, clipped to ±EVAL_CLIP_CP. Mate scored as ±EVAL_CLIP_CP."""
    mate = EVAL_MATE_RE.search(comment)
    if mate:
        return EVAL_CLIP_CP if int(mate.group(1)) > 0 else -EVAL_CLIP_CP
    m = EVAL_NUM_RE.search(comment)
    if not m:
        return None
    cp = int(round(float(m.group(1)) * 100))
    return max(-EVAL_CLIP_CP, min(EVAL_CLIP_CP, cp))


def parse_time_control(tc: str) -> tuple[int, int]:
    if "+" in tc:
        a, b = tc.split("+", 1)
    else:
        a, b = tc, "0"
    try:
        return int(a), int(b)
    except ValueError:
        return 0, 0


def time_control_class(initial: int, increment: int) -> str:
    total = initial + 40 * increment
    if total < 30:
        return "ultrabullet"
    if total < 180:
        return "bullet"
    if total < 480:
        return "blitz"
    if total < 1500:
        return "rapid"
    return "classical"


def material_balance_cp(board: chess.Board, my_color: chess.Color) -> int:
    val = 0
    for pt, v in PIECE_VALUE_CP.items():
        val += v * (len(board.pieces(pt, my_color)) - len(board.pieces(pt, not my_color)))
    return val


def expected_moves_left(num_pcs: int, abs_eval_cp: Optional[int]) -> float:
    base = 5.0 + 1.2 * (num_pcs - 4)
    if abs_eval_cp is None:
        decisiveness = 1.0
    else:
        decisiveness = 1.0 - 0.5 * min(abs_eval_cp, 1000) / 1000.0
    return max(5.0, base * decisiveness)


def detect_phase(num_pcs: int) -> str:
    if num_pcs >= 28:
        return "opening"
    if num_pcs >= 14:
        return "middlegame"
    return "endgame"


def headers_ok(g: chess.pgn.Game, min_year: Optional[int]) -> Optional[str]:
    """Return reason to skip game, or None if game should be processed."""
    if g.headers.get("Variant", "Standard") != "Standard":
        return "non-standard variant"
    white = g.headers.get("White", "")
    black = g.headers.get("Black", "")
    if white.lower() != USER.lower() and black.lower() != USER.lower():
        return "user not in game"
    tc = g.headers.get("TimeControl", "")
    base, _ = parse_time_control(tc)
    if base < MIN_INITIAL_TIME_S:
        return "tc too short"
    if min_year is not None:
        date_s = g.headers.get("UTCDate") or g.headers.get("Date") or ""
        try:
            year = int(date_s.split(".")[0])
        except (ValueError, IndexError):
            return "no parseable date"
        if year < min_year:
            return "too old"
    return None


def extract_game(g: chess.pgn.Game, source: str) -> Iterator[dict]:
    white = g.headers.get("White", "")
    my_color = chess.WHITE if white.lower() == USER.lower() else chess.BLACK
    welo = int(g.headers.get("WhiteElo", "0") or 0)
    belo = int(g.headers.get("BlackElo", "0") or 0)
    my_elo = welo if my_color == chess.WHITE else belo
    opp_elo = belo if my_color == chess.WHITE else welo
    # Elo expected-score formula: depends only on the gap, in [0, 1].
    # Avoids leaking the bot's (unstable) absolute rating into the model at inference.
    if my_elo > 0 and opp_elo > 0:
        elo_expected_score = 1.0 / (1.0 + 10.0 ** ((opp_elo - my_elo) / 400.0))
    else:
        elo_expected_score = 0.5  # missing rating -> assume even

    initial, increment = parse_time_control(g.headers.get("TimeControl", ""))
    tc_class = time_control_class(initial, increment)
    # Lichess: GameId header. Chess.com: extract numeric id from Link.
    game_id = (
        g.headers.get("GameId")
        or g.headers.get("Link", "").rsplit("/", 1)[-1]
        or g.headers.get("Site", "").rsplit("/", 1)[-1]
    )
    date_s = g.headers.get("UTCDate") or g.headers.get("Date") or ""

    board = g.board()
    node = g
    ply = 0

    # Clock state. "_before" = at the start of the player's pending move.
    my_clock_before: float = float(initial)
    opp_clock_before: float = float(initial)
    opp_last_move_time: Optional[float] = None

    prev_eval_cp: Optional[int] = None
    prev_opp_was_capture = False
    prev_opp_to_square: Optional[int] = None

    while node.variations:
        next_node = node.variation(0)
        move = next_node.move
        comment = next_node.comment or ""
        ply += 1

        is_my_move = (board.turn == my_color)
        clock_after = parse_clock(comment)
        eval_cp = parse_eval_cp(comment)
        is_capture = board.is_capture(move)
        to_square = move.to_square

        # Compute time spent on THIS move (works for both sides; only emitted for mine).
        time_spent: Optional[float] = None
        if clock_after is not None:
            clock_before = my_clock_before if is_my_move else opp_clock_before
            time_spent = clock_before + increment - clock_after

        emit = (
            is_my_move
            and ply > SKIP_OPENING_PLIES
            and time_spent is not None
            and time_spent >= MIN_TIME_SPENT
        )

        if emit:
            n_pcs = chess.popcount(board.occupied)
            abs_eval = abs(eval_cp) if eval_cp is not None else None
            recapture = (
                prev_opp_was_capture
                and prev_opp_to_square == to_square
                and is_capture
            )
            row = {
                "source": source,
                "game_id": game_id,
                "date": date_s,
                "time_control_class": tc_class,
                "initial_time": initial,
                "increment": increment,
                "my_color": "white" if my_color == chess.WHITE else "black",
                "elo_expected_score": round(elo_expected_score, 4),
                "ply": ply,
                "move_number": (ply + 1) // 2,
                "phase": detect_phase(n_pcs),
                "num_pieces": n_pcs,
                "material_balance_cp": material_balance_cp(board, my_color),
                "in_check": int(board.is_check()),
                "num_legal_moves": board.legal_moves.count(),
                "is_recapture": int(recapture),
                "expected_moves_left": round(expected_moves_left(n_pcs, abs_eval), 2),
                "eval_cp": eval_cp if eval_cp is not None else "",
                "abs_eval_cp": abs_eval if abs_eval is not None else "",
                "eval_delta": (
                    eval_cp - prev_eval_cp
                    if (eval_cp is not None and prev_eval_cp is not None)
                    else ""
                ),
                "my_time_left": round(my_clock_before, 2),
                "opp_time_left": round(opp_clock_before, 2),
                "my_time_frac": round(my_clock_before / max(initial, 1), 4),
                "time_diff": round(my_clock_before - opp_clock_before, 2),
                "opp_last_move_time": (
                    round(opp_last_move_time, 3) if opp_last_move_time is not None else ""
                ),
                "fen": board.fen(),
                "time_spent": round(time_spent, 3),
            }
            yield row

        # Update state AFTER the move
        if is_my_move:
            if clock_after is not None:
                my_clock_before = clock_after
        else:
            if clock_after is not None:
                opp_last_move_time = time_spent
                opp_clock_before = clock_after
            prev_opp_was_capture = is_capture
            prev_opp_to_square = to_square

        if eval_cp is not None:
            prev_eval_cp = eval_cp

        board.push(move)
        node = next_node


def extract_pgn(
    pgn_path: Path,
    out_path: Path,
    source: str,
    min_year: Optional[int],
    limit: Optional[int],
) -> tuple[int, int]:
    """Stream games from pgn_path, write rows to out_path. Returns (games_kept, rows)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skip_reasons: dict[str, int] = {}
    games_seen = games_kept = rows = 0

    with pgn_path.open("r", encoding="utf-8", errors="replace") as f, \
         out_path.open("w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        while True:
            try:
                game = chess.pgn.read_game(f)
            except Exception as e:
                logger.warning("read_game failed: %s; skipping", e)
                continue
            if game is None:
                break
            games_seen += 1
            if limit is not None and games_seen > limit:
                break

            reason = headers_ok(game, min_year)
            if reason is not None:
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                continue

            game_rows = 0
            try:
                for row in extract_game(game, source):
                    writer.writerow(row)
                    game_rows += 1
            except Exception as e:
                logger.warning("extract failed for game %s: %s", game.headers.get("GameId"), e)
                continue

            if game_rows == 0:
                skip_reasons["no rows (no clocks?)"] = skip_reasons.get("no rows (no clocks?)", 0) + 1
                continue

            games_kept += 1
            rows += game_rows

            if games_seen % 500 == 0:
                logger.info("  seen=%d kept=%d rows=%d", games_seen, games_kept, rows)

    logger.info("Done: seen=%d kept=%d rows=%d -> %s", games_seen, games_kept, rows, out_path)
    if skip_reasons:
        logger.info("Skip reasons:")
        for r, c in sorted(skip_reasons.items(), key=lambda kv: -kv[1]):
            logger.info("  %-30s %d", r, c)
    return games_kept, rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, choices=["chesscom", "lichess"])
    p.add_argument("--pgn", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-year", type=int, default=None,
                   help="Skip games before this UTCDate year (e.g. 2020 for chess.com)")
    p.add_argument("--limit", type=int, default=None, help="Process only first N games (smoke test)")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    extract_pgn(args.pgn, args.out, args.source, args.min_year, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
