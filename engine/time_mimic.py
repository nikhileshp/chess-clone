"""Time-mimicry: predict per-move thinking time, select moves with Maia-dominant softmax.

At inference, the engine:
  1. Computes a feature row for the current position + clock state
  2. Predicts log(time_spent / base_budget) at q=0.25/0.50/0.75 via three LightGBM
     quantile boosters
  3. Samples uniformly across the quantile distribution to get a per-move budget
  4. Allocates that budget across breadth (top-k candidates) and per-candidate
     Stockfish thinking time
  5. Picks a move via Maia-dominant softmax: score = alpha*log(maia_p) + beta*eval/100

Trained models live in weights/time_model/{time_q25.lgb, time_q50.lgb, time_q75.lgb,
meta.json}.
"""
from __future__ import annotations

import json
import logging
import math
import random
import time
from pathlib import Path
from typing import NamedTuple, Optional

import chess
import chess.engine
import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EVAL_CLIP_CP = 1000
TARGET_EVAL_DEPTHS = (4, 6, 8, 10)
LABEL_LOG_MULT_CLAMP = 3.0  # matches training: ~×0.05 .. ×20

# Multiplier safety: caps the model's effect on top of the rule-based base budget.
# (LABEL_LOG_MULT_CLAMP already enforces this in log-space; keep here as a clear floor/ceiling.)
MULT_MIN = math.exp(-LABEL_LOG_MULT_CLAMP)
MULT_MAX = math.exp(LABEL_LOG_MULT_CLAMP)


def expected_moves_left(num_pieces: int, abs_eval_cp: Optional[int]) -> float:
    """Same heuristic used during training — keeps inference consistent."""
    base = 5.0 + 1.2 * (num_pieces - 4)
    if abs_eval_cp is None:
        decisiveness = 1.0
    else:
        decisiveness = 1.0 - 0.5 * min(abs_eval_cp, 1000) / 1000.0
    return max(5.0, base * decisiveness)


def detect_phase(num_pieces: int) -> str:
    if num_pieces >= 28:
        return "opening"
    if num_pieces >= 14:
        return "middlegame"
    return "endgame"


def material_balance_cp(board: chess.Board, my_color: chess.Color) -> int:
    val = 0
    for pt, v in (
        (chess.PAWN, 100),
        (chess.KNIGHT, 300),
        (chess.BISHOP, 300),
        (chess.ROOK, 500),
        (chess.QUEEN, 900),
    ):
        val += v * (len(board.pieces(pt, my_color)) - len(board.pieces(pt, not my_color)))
    return val


def _score_to_cp(score_obj, turn: chess.Color) -> int:
    score = score_obj.pov(turn)
    if score.is_mate():
        return EVAL_CLIP_CP if score.mate() > 0 else -EVAL_CLIP_CP
    cp = score.score(mate_score=EVAL_CLIP_CP)
    return max(-EVAL_CLIP_CP, min(EVAL_CLIP_CP, cp))


def multi_depth_eval(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    target_depths: tuple[int, ...] = TARGET_EVAL_DEPTHS,
) -> dict[int, int]:
    """Single streaming search to max(target_depths); capture eval at each target."""
    max_depth = max(target_depths)
    evals: dict[int, Optional[int]] = {d: None for d in target_depths}
    try:
        with engine.analysis(board, chess.engine.Limit(depth=max_depth)) as analysis:
            for info in analysis:
                if "depth" not in info or "score" not in info:
                    continue
                d = info["depth"]
                if d in evals:
                    evals[d] = _score_to_cp(info["score"], board.turn)
                if d >= max_depth and evals[max_depth] is not None:
                    break
    except Exception as e:  # noqa: BLE001
        logger.warning("multi_depth_eval failed: %s; using zeros", e)
        return {d: 0 for d in target_depths}

    # Forward-fill missing
    last: Optional[int] = None
    for d in sorted(evals):
        if evals[d] is None:
            evals[d] = last
        else:
            last = evals[d]
    if evals[min(target_depths)] is None:
        fallback = next((v for v in evals.values() if v is not None), 0)
        for d in evals:
            if evals[d] is None:
                evals[d] = fallback
    return {d: int(v) for d, v in evals.items()}  # type: ignore[arg-type]


def is_recapture_available(board: chess.Board, prev_opp_capture_sq: Optional[int]) -> bool:
    """Approximation for training's `is_recapture` feature.

    Training defined recapture as "the move I made was a capture on the square
    opponent just captured on". At inference we don't know the move yet, so we
    approximate: opp captured on a square AND we have a legal capturing move
    targeting that square.
    """
    if prev_opp_capture_sq is None:
        return False
    for mv in board.legal_moves:
        if mv.to_square == prev_opp_capture_sq and board.is_capture(mv):
            return True
    return False


class CandidateScore(NamedTuple):
    move: chess.Move
    maia_prob: float
    eval_cp: int
    score: float


class TimeMimic:
    """Loads quantile boosters + meta, builds features, predicts budgets, scores candidates."""

    def __init__(
        self,
        model_dir: Path,
        alpha: float = 1.0,
        beta: float = 0.15,
        temperature: float = 0.3,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

        meta_path = self.model_dir / "meta.json"
        with meta_path.open() as f:
            self.meta = json.load(f)
        self.feature_names: list[str] = self.meta["feature_names"]
        self.cat_features: set[str] = set(self.meta["categorical_features"])

        logger.info("Loading time-mimic boosters from %s", self.model_dir)
        self.q25 = lgb.Booster(model_file=str(self.model_dir / "time_q25.lgb"))
        self.q50 = lgb.Booster(model_file=str(self.model_dir / "time_q50.lgb"))
        self.q75 = lgb.Booster(model_file=str(self.model_dir / "time_q75.lgb"))
        logger.info(
            "Time-mimic ready (alpha=%.2f beta=%.3f T=%.2f, features=%d)",
            alpha, beta, temperature, len(self.feature_names),
        )

    def base_budget(
        self,
        my_time_left: float,
        increment: float,
        expected_moves_left_value: float,
    ) -> float:
        return my_time_left / max(expected_moves_left_value, 1.0) + increment

    def _row_to_df(self, feats: dict) -> pd.DataFrame:
        """Construct a 1-row DataFrame matching the training schema."""
        row = {name: feats.get(name, np.nan) for name in self.feature_names}
        df = pd.DataFrame([row], columns=self.feature_names)
        for col in self.cat_features:
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df

    def predict_log_multiplier(self, feats: dict) -> float:
        """Sample a log-multiplier from the quantile distribution given features."""
        df = self._row_to_df(feats)
        q25 = float(self.q25.predict(df)[0])
        q50 = float(self.q50.predict(df)[0])
        q75 = float(self.q75.predict(df)[0])
        # Enforce monotonicity (quantile crossing is rare but possible)
        q25, q50, q75 = sorted([q25, q50, q75])

        u = random.uniform(0.0, 1.0)
        if u < 0.25:
            # extrapolate below q25 using the q25↔q50 slope
            log_mult = q25 - (q50 - q25) * (1.0 - u / 0.25)
        elif u < 0.50:
            log_mult = q25 + (q50 - q25) * ((u - 0.25) / 0.25)
        elif u < 0.75:
            log_mult = q50 + (q75 - q50) * ((u - 0.50) / 0.25)
        else:
            log_mult = q75 + (q75 - q50) * ((u - 0.75) / 0.25)

        return max(-LABEL_LOG_MULT_CLAMP, min(LABEL_LOG_MULT_CLAMP, log_mult))

    def predict_budget(self, base_budget: float, feats: dict) -> float:
        log_mult = self.predict_log_multiplier(feats)
        return base_budget * math.exp(log_mult)

    def allocate(
        self,
        budget_s: float,
        k_max: int = 5,
        extraction_overhead_s: float = 0.05,
        min_per_cand_s: float = 0.05,
    ) -> tuple[int, float]:
        """budget -> (k_eff, per_candidate_time). k saturates at k_max."""
        k = max(1, min(round(1 + 2.0 * math.log2(1 + budget_s)), k_max))
        per_cand = max(min_per_cand_s, (budget_s - extraction_overhead_s) / k)
        return k, per_cand

    def score_candidates(
        self,
        candidates: list[tuple[chess.Move, float, int]],
    ) -> CandidateScore:
        """Maia-dominant softmax over (move, maia_prob, eval_cp); samples one."""
        if not candidates:
            raise ValueError("no candidates to score")

        scores = []
        for _, p, ev in candidates:
            scores.append(self.alpha * math.log(p + 1e-9) + self.beta * (ev / 100.0))
        scores_np = np.asarray(scores, dtype=np.float64)
        scaled = scores_np / max(self.temperature, 1e-3)
        scaled = scaled - scaled.max()  # for numerical stability
        weights = np.exp(scaled)
        weights = weights / weights.sum()

        idx = int(np.random.choice(len(candidates), p=weights))
        mv, p, ev = candidates[idx]
        return CandidateScore(move=mv, maia_prob=p, eval_cp=ev, score=float(scores[idx]))


def safe_sleep(target_s: float, my_time_left: float, max_sleep_s: float, safety_margin_s: float) -> float:
    """Sleep up to target_s, but never within safety_margin of flagging.

    Returns the actual seconds slept.
    """
    if target_s <= 0:
        return 0.0
    headroom = my_time_left - safety_margin_s
    capped = min(target_s, max_sleep_s, max(0.0, headroom))
    if capped <= 0.05:
        return 0.0
    time.sleep(capped)
    return capped
