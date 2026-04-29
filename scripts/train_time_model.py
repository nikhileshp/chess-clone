"""Train LightGBM quantile-regression models predicting log-time-multiplier.

Target: log(time_spent / base_budget), where base_budget is the rule-based
allocator (my_time_left / expected_moves_left + increment). At inference, the
bot multiplies base_budget by exp(predicted_log_multiplier) to get the
human-mimicking time budget for the next move.

Trains three models (q=0.25, 0.50, 0.75) so inference can sample from a
predicted distribution rather than a point estimate — gives realistic
variability ("I don't always think exactly the same time on the same kind of
position").

Usage:
    uv run python scripts/train_time_model.py \\
        --in data/processed/time_features_all_eval.csv \\
        --out-dir weights/time_model
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

QUANTILES = (0.25, 0.50, 0.75)
CATEGORICAL_FEATURES = ["source", "time_control_class", "my_color", "phase"]

# Columns dropped from the feature matrix:
# - identifiers / leakage:  game_id, date, fen
# - target / derived:       time_spent, log_multiplier, base_budget
EXCLUDED_FROM_FEATURES = [
    "game_id", "date", "fen",
    "time_spent", "log_multiplier", "base_budget",
]

# Sanity bounds on the log-multiplier label.
# exp(±3) ≈ ×0.05 .. ×20 — anything beyond is almost certainly clock anomaly.
LABEL_CLIP = 3.0


def compute_base_budget(df: pd.DataFrame) -> pd.Series:
    """Rule-based allocator: my_time_left/expected_moves_left + increment."""
    return df["my_time_left"] / df["expected_moves_left"] + df["increment"]


def split_by_game(
    df: pd.DataFrame, test_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into train/test by game_id to avoid same-game leakage."""
    rng = np.random.default_rng(seed)
    games = df["game_id"].unique()
    rng.shuffle(games)
    n_test = max(1, int(len(games) * test_fraction))
    test_games = set(games[:n_test])
    test_mask = df["game_id"].isin(test_games)
    return df[~test_mask].copy(), df[test_mask].copy()


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop excluded cols, set categoricals to pandas category dtype."""
    feat = df.drop(columns=[c for c in EXCLUDED_FROM_FEATURES if c in df.columns])
    for col in CATEGORICAL_FEATURES:
        if col in feat.columns:
            feat[col] = feat[col].astype("category")
    return feat


def train_quantile_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    alpha: float,
    seed: int,
) -> lgb.Booster:
    params = {
        "objective": "quantile",
        "alpha": alpha,
        "metric": "quantile",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": seed,
    }
    cat_feats = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]
    train_ds = lgb.Dataset(X_train, y_train, categorical_feature=cat_feats)
    val_ds = lgb.Dataset(X_val, y_val, reference=train_ds, categorical_feature=cat_feats)
    model = lgb.train(
        params,
        train_ds,
        num_boost_round=2000,
        valid_sets=[train_ds, val_ds],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(period=100),
        ],
    )
    return model


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    err = y_true - y_pred
    return float(np.mean(np.maximum(alpha * err, (alpha - 1) * err)))


def coverage_at_quantile(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of true values <= predicted (should match alpha if calibrated)."""
    return float(np.mean(y_true <= y_pred))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("weights/time_model"))
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s ...", args.in_path)
    df = pd.read_csv(args.in_path)
    logger.info("  %d rows, %d cols", len(df), df.shape[1])

    # Build label
    df["base_budget"] = compute_base_budget(df)
    df = df[df["base_budget"] > 0].copy()
    df["log_multiplier"] = np.log(df["time_spent"] / df["base_budget"])

    pre_clip = len(df)
    df = df[df["log_multiplier"].between(-LABEL_CLIP, LABEL_CLIP)].copy()
    logger.info("Label range clipped to ±%.1f: %d -> %d rows", LABEL_CLIP, pre_clip, len(df))

    # Split: by game_id, twice (train+val out of train; test held out)
    train_full, test = split_by_game(df, args.test_fraction, args.seed)
    train, val = split_by_game(train_full, args.val_fraction, args.seed + 1)
    logger.info("Splits: train=%d val=%d test=%d", len(train), len(val), len(test))

    X_train = prepare_features(train)
    X_val = prepare_features(val)
    X_test = prepare_features(test)
    y_train = train["log_multiplier"]
    y_val = val["log_multiplier"]
    y_test = test["log_multiplier"].to_numpy()

    metadata = {
        "feature_names": list(X_train.columns),
        "categorical_features": [c for c in CATEGORICAL_FEATURES if c in X_train.columns],
        "label": "log_multiplier",
        "base_budget_formula": "my_time_left / expected_moves_left + increment",
        "label_clip": LABEL_CLIP,
        "quantiles": list(QUANTILES),
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
    }

    test_metrics: dict[str, float] = {}
    for q in QUANTILES:
        logger.info("=== Training q=%.2f ===", q)
        model = train_quantile_model(X_train, y_train, X_val, y_val, q, args.seed)
        out_path = args.out_dir / f"time_q{int(q * 100):02d}.lgb"
        model.save_model(str(out_path))

        y_pred_test = model.predict(X_test)
        pinball = pinball_loss(y_test, y_pred_test, q)
        cov = coverage_at_quantile(y_test, y_pred_test)
        test_metrics[f"q{int(q * 100):02d}"] = {
            "pinball_loss": pinball,
            "coverage": cov,
            "best_iter": model.best_iteration,
        }
        logger.info(
            "  saved %s | test pinball=%.4f coverage=%.3f (target %.2f) best_iter=%d",
            out_path, pinball, cov, q, model.best_iteration,
        )

    # Feature importance from the median model (most representative)
    median_path = args.out_dir / "time_q50.lgb"
    median_model = lgb.Booster(model_file=str(median_path))
    importance = sorted(
        zip(median_model.feature_name(), median_model.feature_importance(importance_type="gain")),
        key=lambda kv: -kv[1],
    )
    metadata["feature_importance_q50_gain"] = [(name, int(imp)) for name, imp in importance]

    metadata["test_metrics"] = test_metrics
    meta_path = args.out_dir / "meta.json"
    with meta_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Wrote %s", meta_path)

    logger.info("Top features by gain (median model):")
    for name, imp in importance[:15]:
        logger.info("  %-25s %d", name, imp)

    return 0


if __name__ == "__main__":
    sys.exit(main())
