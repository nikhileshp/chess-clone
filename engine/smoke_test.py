"""Local smoke test: spawn maia_engine.py as a subprocess, talk UCI to it,
verify it returns a legal best move.

Use this BEFORE deploying to HF Space to catch import / config / book errors.

Usage:
    python engine/smoke_test.py [--checkpoint PATH]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Override MAIA_CHECKPOINT_PATH for this test")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Per-command read timeout (seconds)")
    args = parser.parse_args()

    here = Path(__file__).parent
    env = os.environ.copy()
    if args.checkpoint:
        env["MAIA_CHECKPOINT_PATH"] = str(args.checkpoint)
    env.setdefault("MAIA_BOOK_DIR", str(here / "book"))

    print(f"Spawning maia_engine.py (book_dir={env['MAIA_BOOK_DIR']}, "
          f"ckpt={env.get('MAIA_CHECKPOINT_PATH', '<auto>')})...")
    proc = subprocess.Popen(
        [sys.executable, str(here / "maia_engine.py")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        bufsize=1,
    )

    def send(cmd: str) -> None:
        print(f"  -> {cmd}")
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()

    def expect(token: str, soft: bool = False) -> str | None:
        deadline = time.time() + args.timeout
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                continue
            line = line.rstrip()
            print(f"  <- {line}")
            if token in line:
                return line
        if soft:
            return None
        raise TimeoutError(f"Expected {token!r} within {args.timeout}s")

    try:
        send("uci")
        expect("uciok")
        send("isready")
        expect("readyok")
        send("ucinewgame")
        # Test 1: starting position, should give a book move
        send("position startpos")
        send("go wtime 60000 btime 60000")
        bm1 = expect("bestmove")
        assert bm1 and bm1.startswith("bestmove"), f"Bad bestmove line: {bm1!r}"
        # Test 2: after 1.e4, should give a Caro-ish response if we're playing your side
        send("position startpos moves e2e4")
        send("go wtime 60000 btime 60000")
        bm2 = expect("bestmove")
        # Test 3: a non-book middlegame FEN to force policy net
        send("position fen r1bqk2r/pp2bppp/2n2n2/3pp3/4P3/2NP1N2/PPP2PPP/R1BQ1RK1 w kq - 0 7")
        send("go wtime 60000 btime 60000")
        bm3 = expect("bestmove")
        send("quit")
        proc.wait(timeout=5)
        print("\n--- SMOKE TEST PASSED ---")
        print(f"  startpos -> {bm1.split()[1]}")
        print(f"  after 1.e4 -> {bm2.split()[1]}")
        print(f"  middlegame -> {bm3.split()[1]}")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"\n--- SMOKE TEST FAILED: {e} ---", file=sys.stderr)
        proc.kill()
        # Print collected stderr for diagnosis
        try:
            err = proc.stderr.read()
            if err:
                print("\n=== engine stderr ===", file=sys.stderr)
                print(err, file=sys.stderr)
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
