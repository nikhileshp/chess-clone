#!/usr/bin/env python3
"""UCI engine wrapping Maia2 fine-tuned for nick_p12.

Strategy:
  1. First N plies (default 12): prefer Polyglot opening book lookup. The book
     was built from nick_p12's own games, so this enforces opening repertoire.
  2. After book exits or no book hit: sample from Maia2's top-K policy with
     a temperature. Argmax-only would play the same opening every game; this
     reproduces the natural variability seen in the user's games.

Reads UCI on stdin, writes responses on stdout. Designed to be invoked by
lichess-bot as a configured engine. Standalone-runnable for testing.

Env vars:
  MAIA_CHECKPOINT_PATH    path to fine-tuned .pt (default: ./model/nick_p12.pt)
  MAIA_BOOK_DIR           dir containing white.bin / black.bin (default: ./book)
  MAIA_DEVICE             "cuda" or "cpu" (default: auto-detect)
  MAIA_TEMPERATURE        softmax temperature for top-K sampling fallback
                          when Stockfish unavailable (default: 0.4)
  MAIA_TOPK               number of Maia top moves to consider (default: 5)
  MAIA_BOOK_MAX_PLIES     hard cap on book lookups (default: 30)
  MAIA_MIN_BOOK_WEIGHT    minimum max-entry weight to USE the book at a given
                          position (default: 15, ~10 games seen). Below this,
                          the position isn't really part of the repertoire and
                          we fall through to Maia2.
  MAIA_STOCKFISH_PATH     path to stockfish binary (default: /usr/games/stockfish)
  MAIA_STOCKFISH_DEPTH    depth for evaluating each Maia candidate (default: 10)
  MAIA_STOCKFISH_DISABLE  set to "1" to disable Stockfish blending entirely
                          (degrades to pure-Maia top-K sampling)
  MAIA_ELO_SELF           ELO bucket idx for the bot (default: 10 = >=2000)
  MAIA_ELO_OPPO           ELO bucket idx for opponent (default: 10; can override
                          per-game via UCI option)
"""
from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path

import chess
import chess.engine
import chess.polyglot
import torch

from maia2.utils import board_to_tensor, get_all_possible_moves

logger = logging.getLogger("maia_engine")
logging.basicConfig(
    level=os.environ.get("MAIA_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,  # never write logs to stdout — UCI uses stdout
)


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


CHECKPOINT_PATH = env("MAIA_CHECKPOINT_PATH", "./model/nick_p12.pt")
BOOK_DIR = env("MAIA_BOOK_DIR", "./book")
DEVICE_PREF = env("MAIA_DEVICE", "")
TEMPERATURE = float(env("MAIA_TEMPERATURE", "0.4"))
TOPK = int(env("MAIA_TOPK", "5"))
BOOK_MAX_PLIES = int(env("MAIA_BOOK_MAX_PLIES", "30"))
MIN_BOOK_WEIGHT = int(env("MAIA_MIN_BOOK_WEIGHT", "15"))
ELO_SELF = int(env("MAIA_ELO_SELF", "10"))
STOCKFISH_PATH = env("MAIA_STOCKFISH_PATH", "/usr/games/stockfish")
STOCKFISH_DEPTH = int(env("MAIA_STOCKFISH_DEPTH", "10"))
STOCKFISH_DISABLE = env("MAIA_STOCKFISH_DISABLE", "0") == "1"
ELO_OPPO_DEFAULT = int(env("MAIA_ELO_OPPO", "10"))


def pick_device() -> torch.device:
    if DEVICE_PREF == "cuda":
        return torch.device("cuda")
    if DEVICE_PREF == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MaiaEngine:
    """UCI-speaking wrapper around the fine-tuned Maia2 model + opening books."""

    def __init__(self) -> None:
        self.device = pick_device()
        logger.info("Device: %s", self.device)
        self._move_dict = {m: i for i, m in enumerate(get_all_possible_moves())}
        self._inv_moves = {v: k for k, v in self._move_dict.items()}
        self._board = chess.Board()
        self._elo_oppo = ELO_OPPO_DEFAULT

        from maia2 import model as _model_mod  # noqa: WPS433

        logger.info("Loading pretrained Maia2-blitz...")
        self._model = _model_mod.from_pretrained(
            type="blitz", device="gpu" if self.device.type == "cuda" else "cpu"
        )

        if CHECKPOINT_PATH and Path(CHECKPOINT_PATH).exists():
            logger.info("Loading fine-tuned checkpoint: %s", CHECKPOINT_PATH)
            ckpt = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=False)
            sd = ckpt.get("state_dict", ckpt)
            self._model.load_state_dict(sd)
            logger.info("  prior best val top-1: %s", ckpt.get("val_top1", "?"))
        else:
            logger.warning(
                "No fine-tuned checkpoint at %s; running base Maia2-blitz only", CHECKPOINT_PATH
            )

        self._model.eval()

        self._book_white = self._open_book(Path(BOOK_DIR) / "white.bin")
        self._book_black = self._open_book(Path(BOOK_DIR) / "black.bin")

        # Stockfish for ranking among Maia's top-K candidates.
        # Strict mode: bot can only play moves Maia ranked in top-K, but
        # picks the tactically best one of those (no blunder among style).
        self._stockfish: chess.engine.SimpleEngine | None = None
        if not STOCKFISH_DISABLE:
            try:
                self._stockfish = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
                self._stockfish.configure({"Threads": 1, "Hash": 32})
                logger.info(
                    "Stockfish loaded from %s (depth=%d) for top-%d candidate ranking",
                    STOCKFISH_PATH,
                    STOCKFISH_DEPTH,
                    TOPK,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Stockfish unavailable (%s); falling back to pure-Maia sampling", e
                )
                self._stockfish = None
        else:
            logger.info("Stockfish blending disabled via MAIA_STOCKFISH_DISABLE=1")

    @staticmethod
    def _open_book(path: Path):
        if not path.exists():
            logger.warning("Book missing: %s", path)
            return None
        logger.info("Loaded opening book: %s (%.1f KB)", path, path.stat().st_size / 1024)
        return chess.polyglot.MemoryMappedReader(str(path))

    # --- UCI command handlers -------------------------------------------------

    def cmd_uci(self) -> None:
        print("id name nick_p12_bot")
        print("id author Nikhilesh Prabhakar (chess account predates Danny Rensch's; ask Magnus, he's a friend)")
        # Custom options: opponent rating, sampling temperature
        print("option name OpponentElo type spin default 2000 min 800 max 3000")
        print("option name Temperature type string default 0.4")
        print("option name TopK type spin default 5 min 1 max 20")
        print("uciok")
        sys.stdout.flush()

    def cmd_isready(self) -> None:
        print("readyok")
        sys.stdout.flush()

    def cmd_setoption(self, line: str) -> None:
        # "setoption name X value Y"
        parts = line.split()
        if "name" not in parts or "value" not in parts:
            return
        name = parts[parts.index("name") + 1]
        value = " ".join(parts[parts.index("value") + 1:])
        global TEMPERATURE, TOPK  # noqa: WPS420 — ok for env-derived knobs
        if name == "OpponentElo":
            try:
                raw = int(value)
                self._elo_oppo = self._elo_to_bucket(raw)
                logger.info("Opponent rating %d -> bucket %d", raw, self._elo_oppo)
            except ValueError:
                pass
        elif name == "Temperature":
            try:
                TEMPERATURE = float(value)
            except ValueError:
                pass
        elif name == "TopK":
            try:
                TOPK = int(value)
            except ValueError:
                pass

    def cmd_ucinewgame(self) -> None:
        self._board = chess.Board()
        self._elo_oppo = ELO_OPPO_DEFAULT

    def cmd_position(self, line: str) -> None:
        # "position [startpos | fen <fen>] [moves m1 m2 ...]"
        rest = line[len("position"):].strip()
        if rest.startswith("startpos"):
            self._board = chess.Board()
            rest = rest[len("startpos"):].strip()
        elif rest.startswith("fen"):
            rest = rest[len("fen"):].strip()
            # Find moves keyword to split fen from move list
            moves_idx = rest.find(" moves ")
            if moves_idx == -1:
                fen_str, rest = rest, ""
            else:
                fen_str = rest[:moves_idx]
                rest = rest[moves_idx + len(" moves "):]
            self._board = chess.Board(fen_str.strip())
            rest = "moves " + rest if rest else ""
        if rest.startswith("moves"):
            for mv in rest[len("moves"):].split():
                try:
                    self._board.push_uci(mv)
                except ValueError:
                    logger.error("Bad move in position cmd: %s", mv)

    def cmd_go(self, _line: str) -> None:
        move = self.choose_move()
        print(f"bestmove {move.uci()}")
        sys.stdout.flush()

    # --- Move choice ----------------------------------------------------------

    def choose_move(self) -> chess.Move:
        ply = self._board.ply()
        legal = list(self._board.legal_moves)
        if not legal:
            # Should never happen in practice — UCI engines aren't asked to move
            # in mate/stalemate. Return a null-ish move; lichess-bot will resign.
            return chess.Move.null()

        # 1) Try opening book — but only if the position is actually part of
        #    the user's repertoire (max entry weight >= threshold). This lets
        #    the book go deep into mainlines (Caro, QGD, etc.) while falling
        #    through to Maia2 in unfamiliar branches the user only saw once.
        if ply < BOOK_MAX_PLIES:
            book_move = self._book_pick()
            if book_move is not None:
                logger.info("ply=%d book -> %s", ply, book_move.uci())
                return book_move

        # 2) Maia2 policy
        return self._policy_pick(legal)

    def _book_pick(self) -> chess.Move | None:
        is_white = self._board.turn == chess.WHITE
        book = self._book_white if is_white else self._book_black
        if book is None:
            return None
        try:
            entries = list(book.find_all(self._board))
        except Exception as e:  # noqa: BLE001
            logger.warning("Book error: %s", e)
            return None
        if not entries:
            return None

        # Frequency gate: only use the book if the user has played this position
        # often enough that we trust the distribution. Weights are absolute
        # (sum of per-game result_weights), so weight ~ game count * 1-2.
        # Threshold 15 ~= 10+ games seen.
        max_w = max(e.weight for e in entries)
        if max_w < MIN_BOOK_WEIGHT:
            logger.debug(
                "ply=%d skip book (max_w=%d < %d)", self._board.ply(), max_w, MIN_BOOK_WEIGHT
            )
            return None

        # Filter to legal (Polyglot can produce illegal moves on rare malformed books)
        legal_set = set(self._board.legal_moves)
        pairs = [(e.move, e.weight) for e in entries if e.move in legal_set]
        if not pairs:
            return None
        # Drop entries below threshold relative to the dominant move so we
        # don't sample a 1-game oddity when there's a 200-game main line.
        # Relative threshold: keep moves whose weight >= 5% of max.
        rel_cutoff = max(1, max_w // 20)
        pairs = [(m, w) for m, w in pairs if w >= rel_cutoff] or pairs
        moves, weights = zip(*pairs)
        return random.choices(moves, weights=weights, k=1)[0]

    def _policy_pick(self, legal: list[chess.Move]) -> chess.Move:
        x = board_to_tensor(self._board).unsqueeze(0).to(self.device)
        es = torch.tensor([ELO_SELF], device=self.device)
        eo = torch.tensor([self._elo_oppo], device=self.device)
        with torch.no_grad():
            out = self._model(x, es, eo)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        logits = logits.squeeze(0).cpu()

        # Mask illegal moves to -inf so they cannot be sampled
        legal_idxs = [self._move_dict[m.uci()] for m in legal if m.uci() in self._move_dict]
        if not legal_idxs:
            logger.warning("No legal move in action space; falling back to random legal")
            return random.choice(legal)
        mask = torch.full_like(logits, float("-inf"))
        mask[legal_idxs] = 0.0
        masked = logits + mask

        # Take Maia's top-K candidates
        k = min(TOPK, len(legal_idxs))
        top_vals, top_inds = torch.topk(masked, k)
        probs = torch.softmax(top_vals / max(TEMPERATURE, 1e-3), dim=-1)
        candidates: list[tuple[chess.Move, float]] = []  # (move, maia_prob)
        for j in range(k):
            uci = self._inv_moves[top_inds[j].item()]
            candidates.append((chess.Move.from_uci(uci), float(probs[j].item())))

        # Strict-style Stockfish ranking: pick the highest-evaluated move
        # AMONG these candidates. Never plays a move outside Maia's top-K.
        if self._stockfish is not None:
            return self._stockfish_rank(candidates)

        # Fallback: pure-Maia sampling (when Stockfish unavailable)
        weights = [p for _, p in candidates]
        chosen = random.choices([m for m, _ in candidates], weights=weights, k=1)[0]
        logger.info("ply=%d policy[no-sf] -> %s", self._board.ply(), chosen.uci())
        return chosen

    def _stockfish_rank(
        self, candidates: list[tuple[chess.Move, float]]
    ) -> chess.Move:
        """Score each Maia candidate with Stockfish, return the best.

        Strict mode: only candidates from Maia's top-K are considered. Never
        introduces a move outside the user's style distribution.
        """
        my_color = self._board.turn
        scored: list[tuple[float, chess.Move, float]] = []  # (sf_score, move, maia_p)
        for mv, p in candidates:
            self._board.push(mv)
            try:
                info = self._stockfish.analyse(
                    self._board, chess.engine.Limit(depth=STOCKFISH_DEPTH)
                )
                pov = info["score"].pov(my_color)
                # Convert PovScore to centipawns (mate = ±100k)
                cp = pov.score(mate_score=100000)
                if cp is None:
                    cp = -100000  # treat unscoreable as worst
            except Exception as e:  # noqa: BLE001
                logger.warning("Stockfish eval failed on %s: %s", mv.uci(), e)
                cp = -100000
            self._board.pop()
            scored.append((float(cp), mv, p))

        # Pick the candidate with the best Stockfish eval; break ties by Maia prob.
        scored.sort(key=lambda t: (-t[0], -t[2]))
        best_cp, best_mv, best_p = scored[0]
        worst_cp = scored[-1][0]
        logger.info(
            "ply=%d policy+sf -> %s (sf_cp=%+d, maia_p=%.3f, span=%dcp across top-%d)",
            self._board.ply(),
            best_mv.uci(),
            int(best_cp),
            best_p,
            int(best_cp - worst_cp),
            len(scored),
        )
        return best_mv

    @staticmethod
    def _elo_to_bucket(elo: int) -> int:
        """Match Maia2's map_to_category: 11 buckets, 0..10."""
        if elo < 1100:
            return 0
        if elo >= 2000:
            return 10
        return ((elo - 1100) // 100) + 1


def main() -> int:
    engine = MaiaEngine()
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            if line == "uci":
                engine.cmd_uci()
            elif line == "isready":
                engine.cmd_isready()
            elif line.startswith("setoption"):
                engine.cmd_setoption(line)
            elif line == "ucinewgame":
                engine.cmd_ucinewgame()
            elif line.startswith("position"):
                engine.cmd_position(line)
            elif line.startswith("go"):
                engine.cmd_go(line)
            elif line == "quit":
                if engine._stockfish is not None:
                    try:
                        engine._stockfish.quit()
                    except Exception:  # noqa: BLE001
                        pass
                return 0
            elif line == "stop":
                pass  # we don't do iterative search; nothing to stop
            else:
                logger.debug("Unknown UCI: %s", line)
        except Exception as e:  # noqa: BLE001
            logger.exception("Error handling UCI line %r: %s", line, e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
