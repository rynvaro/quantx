"""
Training orchestration pipeline.

Flow
----
1. Load features.parquet, apply eligibility filters
2. Generate walk-forward folds
3. Per fold: train 3 heads, collect OOF predictions, evaluate
4. Fit Platt calibrator on all OOF predictions
5. Retrain final models on full eligible dataset
6. Save artifacts

Main entry: run()
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .fold import walk_forward_folds
from .model import train_fold, fit_calibrator, save_models
from .evaluate import evaluate_fold, summarise_folds

log = logging.getLogger(__name__)

FEATURE_COLS_EXCLUDE = {
    "y_class", "y_max", "y_min",
    "is_weekend", "is_black_swan",
    # OHLC columns embedded in parquet for backtest use only — not available at inference
    "price_open", "price_high", "price_low", "price_close",
}


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _load_and_filter(parquet_path: Path) -> pd.DataFrame:
    """Load parquet and apply eligibility mask."""
    log.info("Loading %s …", parquet_path)
    df = pd.read_parquet(parquet_path)
    log.info("  Raw rows: %d", len(df))

    mask = (
        (df["is_weekend"] == 0)
        & (df["is_black_swan"] == 0)
        & df["y_class"].notna()
        & df["y_max"].notna()
        & df["y_min"].notna()
    )
    df = df[mask].copy()
    log.info("  Eligible rows: %d  (pos=%.1f%%)", len(df), df["y_class"].mean() * 100)
    return df


def _feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return (X, feature_names) from a filtered DataFrame."""
    feat_cols = [c for c in df.columns if c not in FEATURE_COLS_EXCLUDE]
    X = df[feat_cols].values.astype(np.float32)
    return X, feat_cols


# ─── Main entry point ────────────────────────────────────────────────────────


def run(
    parquet_path: Path,
    output_dir: Path,
    n_folds: int = 6,
    val_months: int = 4,
    gap_days: int = 5,
) -> None:
    """
    Full training pipeline.

    Parameters
    ----------
    parquet_path : path to features parquet file.
    output_dir   : directory where model artifacts are saved.
    n_folds      : number of walk-forward folds.
    val_months   : validation window size in months per fold.
    gap_days     : days gap between train end and val start.
    """
    t0 = time.time()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    df = _load_and_filter(parquet_path)
    X, feat_cols = _feature_matrix(df)
    log.info("Feature matrix: %d rows × %d columns", *X.shape)

    y_cls = df["y_class"].values.astype(np.float32)
    y_max = df["y_max"].values.astype(np.float32)
    y_min = df["y_min"].values.astype(np.float32)

    # ── 2. Walk-forward folds ─────────────────────────────────────────────────
    folds = list(walk_forward_folds(df, n_folds=n_folds, val_months=val_months, gap_days=gap_days))
    if not folds:
        raise RuntimeError("No valid folds generated. Check data date range and fold parameters.")

    # OOF containers (classifier only; regressors only evaluated on pos-class)
    oof_scores = np.full(len(df), np.nan, dtype=np.float64)
    oof_labels = y_cls.copy()

    fold_metrics = []

    # ── 3. Per-fold training + evaluation ────────────────────────────────────
    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        log.info("── Fold %d/%d ────────────────────────────────", fold_idx + 1, len(folds))

        X_tr, X_va = X[tr_idx], X[va_idx]
        y_cls_tr, y_cls_va = y_cls[tr_idx], y_cls[va_idx]
        y_max_tr, y_max_va = y_max[tr_idx], y_max[va_idx]
        y_min_tr, y_min_va = y_min[tr_idx], y_min[va_idx]

        model_cls, model_max, model_min = train_fold(
            X_tr, y_cls_tr, y_max_tr, y_min_tr,
            X_va, y_cls_va, y_max_va, y_min_va,
        )

        # OOF classifier scores (raw, before calibration)
        raw_scores = model_cls.predict_proba(X_va)[:, 1]
        oof_scores[va_idx] = raw_scores

        # Predictions for regression heads (now on ALL validation samples)
        y_max_pred = model_max.predict(X_va)
        y_min_pred = model_min.predict(X_va)

        # Evaluate (use raw scores — calibrator not fitted yet, so use as-is for ranking)
        metrics = evaluate_fold(
            y_cls_true=y_cls_va,
            y_cls_score=raw_scores,
            y_max_true=y_max_va,
            y_max_pred=y_max_pred,
            y_min_true=y_min_va,
            y_min_pred=y_min_pred,
            fold=fold_idx + 1,
        )
        fold_metrics.append(metrics)

    # ── 4. Platt Scaling calibrator ───────────────────────────────────────────
    log.info("── Fitting Platt calibrator ────────────────────────────")
    valid_oof = ~np.isnan(oof_scores)
    calibrator = fit_calibrator(oof_scores[valid_oof], oof_labels[valid_oof])

    # ── 5. Print evaluation summary ───────────────────────────────────────────
    summarise_folds(fold_metrics)

    # ── 6. Retrain final models on full data ──────────────────────────────────
    log.info("── Retraining final models on full dataset ─────────────")
    # Use last fold's val as a stand-in eval set for early stopping
    _, last_va_idx = folds[-1]
    X_last_va    = X[last_va_idx]
    y_cls_last   = y_cls[last_va_idx]
    y_max_last   = y_max[last_va_idx]
    y_min_last   = y_min[last_va_idx]

    final_cls, final_max, final_min = train_fold(
        X, y_cls, y_max, y_min,
        X_last_va, y_cls_last, y_max_last, y_min_last,
    )

    # ── 7. Save artifacts ─────────────────────────────────────────────────────
    save_models(final_cls, final_max, final_min, calibrator, output_dir)

    # Save feature column list alongside models (needed for inference)
    feat_col_path = output_dir / "feature_cols.txt"
    feat_col_path.write_text("\n".join(feat_cols))
    log.info("Feature column list saved to %s", feat_col_path)

    elapsed = time.time() - t0
    log.info("Training complete in %.1f s", elapsed)
