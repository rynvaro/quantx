"""
Evaluation metrics for walk-forward folds.

Computes:
  Classification : AUC-PR, AUC-ROC, and Precision/Recall/F1 at multiple thresholds
  Regression     : MAE and Pearson r for y_max and y_min heads
  Business       : Expected net profit per trade at each operating threshold
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support,
    mean_absolute_error,
)
from scipy.stats import pearsonr

log = logging.getLogger(__name__)

# Operating thresholds to evaluate (on calibrated probabilities)
EVAL_THRESHOLDS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]

# Business metric constants (matching training data parameters)
NOTIONAL      = 5850.0   # 78U margin × 75x leverage
FEE           = NOTIONAL * 0.0008   # 4.68U double-sided taker fee
CAPTURE_RATE  = 0.65     # fraction of y_max achievable with market orders
RW_LOSS       = NOTIONAL * 0.001    # ~5.85U random-walk loss for false positives


def evaluate_fold(
    y_cls_true: np.ndarray,
    y_cls_score: np.ndarray,       # calibrated probability from classifier
    y_max_true: np.ndarray,        # only positive-class samples
    y_max_pred: np.ndarray,        # only positive-class samples
    y_min_true: np.ndarray,        # only positive-class samples
    y_min_pred: np.ndarray,        # only positive-class samples
    fold: int = 0,
) -> dict:
    """
    Compute all metrics for a single fold's validation set.

    Parameters
    ----------
    y_cls_true  : true binary labels (0/1), shape (n,)
    y_cls_score : calibrated predicted probabilities, shape (n,)
    y_max_true  : true y_max values for positive-class rows, shape (p,)
    y_max_pred  : predicted y_max values, shape (p,)
    y_min_true  : true y_min values for positive-class rows, shape (p,)
    y_min_pred  : predicted y_min values, shape (p,)
    fold        : fold index for logging

    Returns
    -------
    dict with all metric values (flat, suitable for DataFrame row).
    """
    result: dict = {"fold": fold}

    # ── Classification ────────────────────────────────────────────────────────
    result["auc_pr"]  = average_precision_score(y_cls_true, y_cls_score)
    result["auc_roc"] = roc_auc_score(y_cls_true, y_cls_score)

    for thr in EVAL_THRESHOLDS:
        y_pred_bin = (y_cls_score >= thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_cls_true, y_pred_bin, pos_label=1, average="binary", zero_division=0,
        )
        n_signals = int(y_pred_bin.sum())

        result[f"prec@{thr:.2f}"] = round(prec, 4)
        result[f"rec@{thr:.2f}"]  = round(rec,  4)
        result[f"f1@{thr:.2f}"]   = round(f1,   4)
        result[f"n@{thr:.2f}"]    = n_signals

        # Business: expected net PnL per trade
        if prec > 0 and n_signals > 0:
            tp_gain  = NOTIONAL * np.mean(y_max_true) * CAPTURE_RATE - FEE
            fp_loss  = -(FEE + RW_LOSS)
            exp_pnl  = prec * tp_gain + (1 - prec) * fp_loss
        else:
            exp_pnl = np.nan
        result[f"exp_pnl@{thr:.2f}"] = round(exp_pnl, 2) if not np.isnan(exp_pnl) else np.nan

    # ── Regression: y_max ─────────────────────────────────────────────────────
    if len(y_max_true) > 1:
        result["mae_ymax"] = round(mean_absolute_error(y_max_true, y_max_pred) * 100, 4)  # in %
        result["r_ymax"]   = round(float(pearsonr(y_max_true, y_max_pred).statistic), 4)
    else:
        result["mae_ymax"] = np.nan
        result["r_ymax"]   = np.nan

    # ── Regression: y_min ─────────────────────────────────────────────────────
    if len(y_min_true) > 1:
        result["mae_ymin"] = round(mean_absolute_error(y_min_true, y_min_pred) * 100, 4)
        result["r_ymin"]   = round(float(pearsonr(y_min_true, y_min_pred).statistic), 4)
    else:
        result["mae_ymin"] = np.nan
        result["r_ymin"]   = np.nan

    return result


def summarise_folds(fold_metrics: list[dict]) -> None:
    """
    Print a formatted summary table across all folds.

    Parameters
    ----------
    fold_metrics : list of dicts returned by evaluate_fold().
    """
    df = pd.DataFrame(fold_metrics).set_index("fold")

    print("\n" + "=" * 70)
    print("WALK-FORWARD EVALUATION SUMMARY")
    print("=" * 70)

    # Core classification metrics
    cls_cols = ["auc_pr", "auc_roc"]
    print("\n── Classification (per fold) ──────────────────────────────")
    print(df[cls_cols].to_string(float_format="{:.4f}".format))
    print(f"\n  auc_pr  mean={df['auc_pr'].mean():.4f}  std={df['auc_pr'].std():.4f}")
    print(f"  auc_roc mean={df['auc_roc'].mean():.4f}  std={df['auc_roc'].std():.4f}")

    # Best operating threshold (maximise expected PnL)
    print("\n── Business metrics at each operating threshold ───────────")
    pnl_cols  = [c for c in df.columns if c.startswith("exp_pnl@")]
    prec_cols = [c for c in df.columns if c.startswith("prec@")]
    rec_cols  = [c for c in df.columns if c.startswith("rec@")]
    n_cols    = [c for c in df.columns if c.startswith("n@")]

    header = f"{'thr':>6}  {'exp_pnl/trade':>14}  {'prec':>6}  {'rec':>6}  {'signals/fold':>13}"
    print(header)
    print("-" * len(header))
    for pnl_c, prec_c, rec_c, n_c in zip(pnl_cols, prec_cols, rec_cols, n_cols):
        thr = pnl_c.split("@")[1]
        print(
            f"  {thr:>4}  "
            f"{df[pnl_c].mean():>12.2f}U  "
            f"{df[prec_c].mean():>6.3f}  "
            f"{df[rec_c].mean():>6.3f}  "
            f"{df[n_c].mean():>11.0f}"
        )

    # Regression metrics
    print("\n── Regression (per fold) ──────────────────────────────────")
    reg_cols = ["mae_ymax", "r_ymax", "mae_ymin", "r_ymin"]
    print(df[reg_cols].to_string(float_format="{:.4f}".format))
    print(
        f"\n  mae_ymax mean={df['mae_ymax'].mean():.4f}%  "
        f"r_ymax mean={df['r_ymax'].mean():.4f}"
    )
    print(
        f"  mae_ymin mean={df['mae_ymin'].mean():.4f}%  "
        f"r_ymin mean={df['r_ymin'].mean():.4f}"
    )
    print("=" * 70 + "\n")
