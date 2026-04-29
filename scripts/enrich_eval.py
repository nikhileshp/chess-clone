"""Backfill `eval_cp`, `abs_eval_cp`, `eval_delta` columns in a time-features CSV
by running a low-depth Stockfish on each row's FEN.

Reasons we don't trust source `%eval`:
- Chess.com PGNs have no eval at all.
- Lichess `%eval` is at depth ~22, but our bot uses depth 10 at inference.
  Training on depth-22 features and inferring on depth-10 introduces
  distribution shift; we want train-time features to match inference.

Usage:
    uv run python scripts/enrich_eval.py \\
        --in data/processed/time_features_chesscom.csv \\
        --out data/processed/time_features_chesscom_eval.csv \\
        --depth 10 --workers 8

For a quick smoke run, add `--sample 1000` and `--workers 2`.
"""
from __future__ import annotations

import argparse
import csv
import logging
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

import chess
import chess.engine

logger = logging.getLogger(__name__)

EVAL_CLIP_CP = 1000
TARGET_DEPTHS = (4, 6, 8, 10)  # captured during a single iterative-deepening search
DEFAULT_STOCKFISH = os.environ.get("MAIA_STOCKFISH_PATH", "/usr/games/stockfish")


def _score_to_cp(score_obj, turn: chess.Color) -> int:
    score = score_obj.pov(turn)
    if score.is_mate():
        return EVAL_CLIP_CP if score.mate() > 0 else -EVAL_CLIP_CP
    cp = score.score(mate_score=EVAL_CLIP_CP)
    return max(-EVAL_CLIP_CP, min(EVAL_CLIP_CP, cp))


def evaluate_position(
    engine: chess.engine.SimpleEngine, fen: str, target_depths: tuple[int, ...] = TARGET_DEPTHS
) -> dict[int, int]:
    """Return {depth: cp_eval} for each target depth via a single streaming search.

    Iterative deepening means a single Limit(depth=max(target_depths)) search visits
    every shallower depth on the way up; we capture the final score at each.
    Forward-fills any missing depth with the next-shallower value as a safety net.
    """
    board = chess.Board(fen)
    max_depth = max(target_depths)
    evals: dict[int, Optional[int]] = {d: None for d in target_depths}

    with engine.analysis(board, chess.engine.Limit(depth=max_depth)) as analysis:
        for info in analysis:
            if "depth" not in info or "score" not in info:
                continue
            d = info["depth"]
            if d in evals:
                evals[d] = _score_to_cp(info["score"], board.turn)
            if d >= max_depth and evals[max_depth] is not None:
                break

    # Forward-fill missing depths (rare — Stockfish skipped past them too fast for analysis() to emit).
    last: Optional[int] = None
    for d in sorted(evals):
        if evals[d] is None:
            evals[d] = last
        else:
            last = evals[d]
    # If the shallowest is still None, backfill from any available value.
    if evals[min(target_depths)] is None:
        fallback = next((v for v in evals.values() if v is not None), 0)
        for d in evals:
            if evals[d] is None:
                evals[d] = fallback
    return {d: int(v) for d, v in evals.items()}  # type: ignore[arg-type]


def worker_process(
    worker_id: int,
    in_queue: "mp.Queue[Optional[tuple[int, str]]]",
    out_queue: "mp.Queue[tuple[int, dict[int, int]]]",
    sf_path: str,
    target_depths: tuple[int, ...],
) -> None:
    """Worker: pulls (row_idx, fen), pushes (row_idx, {depth: cp_eval})."""
    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    engine.configure({"Threads": 1, "Hash": 32})
    try:
        while True:
            item = in_queue.get()
            if item is None:
                break
            idx, fen = item
            try:
                evals = evaluate_position(engine, fen, target_depths)
            except Exception as e:
                logger.warning("worker %d: eval failed on idx=%d: %s", worker_id, idx, e)
                evals = {d: 0 for d in target_depths}  # neutral fallback
            out_queue.put((idx, evals))
    finally:
        engine.quit()


def stream_eval_results(
    rows: list[dict],
    sf_path: str,
    target_depths: tuple[int, ...],
    workers: int,
) -> dict[int, dict[int, int]]:
    """Run multiprocessing pool of Stockfish workers; return {row_idx: {depth: cp_eval}}."""
    in_queue: "mp.Queue[Optional[tuple[int, str]]]" = mp.Queue(maxsize=4 * workers)
    out_queue: "mp.Queue[tuple[int, dict[int, int]]]" = mp.Queue()

    procs = []
    for w in range(workers):
        p = mp.Process(
            target=worker_process,
            args=(w, in_queue, out_queue, sf_path, target_depths),
        )
        p.daemon = True
        p.start()
        procs.append(p)

    total = len(rows)
    results: dict[int, dict[int, int]] = {}
    started = time.monotonic()

    # Feeder: enqueue all jobs, then poison-pill each worker
    def feed() -> None:
        for idx, row in enumerate(rows):
            in_queue.put((idx, row["fen"]))
        for _ in range(workers):
            in_queue.put(None)

    feeder = mp.Process(target=feed, daemon=True)
    feeder.start()

    last_log = started
    while len(results) < total:
        idx, evals = out_queue.get()
        results[idx] = evals
        now = time.monotonic()
        if now - last_log > 5.0:
            done = len(results)
            rate = done / (now - started)
            eta = (total - done) / max(rate, 1e-6)
            logger.info("  %d/%d (%.0f rows/s, eta %.0fs)", done, total, rate, eta)
            last_log = now

    feeder.join(timeout=5)
    for p in procs:
        p.join(timeout=5)

    return results


def enrich_csv(
    in_path: Path,
    out_path: Path,
    sf_path: str,
    target_depths: tuple[int, ...],
    workers: int,
    sample: Optional[int],
    seed: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    primary_depth = max(target_depths)

    logger.info("Loading %s ...", in_path)
    with in_path.open("r") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    logger.info("  %d rows", len(rows))

    # Add per-depth columns alongside the existing eval_cp (which we set to depth-10).
    new_cols = [f"eval_cp_d{d}" for d in target_depths if d != primary_depth]
    output_fields = fieldnames + [c for c in new_cols if c not in fieldnames]

    if sample is not None and sample < len(rows):
        random.seed(seed)
        rows = random.sample(rows, sample)
        logger.info("  sampled to %d rows", len(rows))

    logger.info(
        "Evaluating with stockfish=%s depths=%s workers=%d",
        sf_path, list(target_depths), workers,
    )
    started = time.monotonic()
    results = stream_eval_results(rows, sf_path, target_depths, workers)
    elapsed = time.monotonic() - started
    logger.info("Eval pass done in %.1fs (%.0f rows/s)", elapsed, len(rows) / max(elapsed, 1e-6))

    rows_with_eval = []
    for idx, row in enumerate(rows):
        evals = results[idx]
        primary_cp = evals[primary_depth]
        new_row = dict(row)
        new_row["eval_cp"] = primary_cp
        new_row["abs_eval_cp"] = abs(primary_cp)
        for d in target_depths:
            if d != primary_depth:
                new_row[f"eval_cp_d{d}"] = evals[d]
        rows_with_eval.append(new_row)

    # Recompute eval_delta within each game (in ply order) using primary-depth eval.
    # When --sample is used, ply-continuity within a game is broken, so leave delta blank.
    if sample is None:
        rows_with_eval.sort(key=lambda r: (r["game_id"], int(r["ply"])))
        prev_game = None
        prev_eval = 0
        for row in rows_with_eval:
            gid = row["game_id"]
            if gid != prev_game:
                row["eval_delta"] = ""
                prev_game = gid
            else:
                row["eval_delta"] = int(row["eval_cp"]) - prev_eval
            prev_eval = int(row["eval_cp"])
    else:
        for row in rows_with_eval:
            row["eval_delta"] = ""

    logger.info("Writing %s (%d cols)", out_path, len(output_fields))
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        for row in rows_with_eval:
            writer.writerow(row)
    logger.info("Done.")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", type=Path, required=True)
    p.add_argument("--out", dest="out_path", type=Path, required=True)
    p.add_argument("--depths", type=int, nargs="+", default=list(TARGET_DEPTHS),
                   help="Target depths to capture (deepest is also written to eval_cp).")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    p.add_argument("--sample", type=int, default=None,
                   help="Randomly sample N rows for faster iteration")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stockfish", default=DEFAULT_STOCKFISH)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not Path(args.stockfish).exists():
        logger.error("Stockfish not found at %s. Install it (apt install stockfish) "
                     "or set --stockfish PATH.", args.stockfish)
        return 1

    enrich_csv(
        args.in_path, args.out_path, args.stockfish,
        tuple(sorted(args.depths)), args.workers, args.sample, args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
