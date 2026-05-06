"""
3-Head model training — supports XGBoost (default) and LightGBM.

Heads
-----
  cls : Classifier  → y_class  (binary, all eligible samples)
  max : Regressor   → y_max    (continuous, all samples)
  min : Regressor   → y_min    (continuous, all samples)

After walk-forward OOF collection, a Platt Scaling calibrator is fitted on
the classifier's OOF scores to produce calibrated probabilities.

Backend
-------
  xgb (default) : XGBoost — saves as model_cls.json / model_max.json / model_min.json
  lgb           : LightGBM — saves as model_cls.lgb  / model_max.lgb  / model_min.lgb
"""

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor
import lightgbm as lgb

log = logging.getLogger(__name__)

# ─── Default hyperparameters ──────────────────────────────────────────────────

_CLS_PARAMS: dict[str, Any] = {
    "n_estimators":      500,
    "max_depth":         6,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  5,
    "gamma":             0.1,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "objective":         "binary:logistic",
    "eval_metric":       "logloss",
    "tree_method":       "hist",
    "random_state":      42,
    "n_jobs":            -1,
}

_REG_PARAMS: dict[str, Any] = {
    "n_estimators":      500,
    "max_depth":         5,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  5,
    "gamma":             0.05,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "objective":         "reg:squarederror",
    "eval_metric":       "mae",
    "tree_method":       "hist",
    "random_state":      42,
    "n_jobs":            -1,
}

_EARLY_STOPPING = 50
_VERBOSE_EVAL   = False  # set True for per-round loss printout

_LGB_CLS_PARAMS: dict[str, Any] = {
    "n_estimators":      500,
    "max_depth":         6,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  5,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "objective":         "binary",
    "metric":            "binary_logloss",
    "random_state":      42,
    "n_jobs":            -1,
    "verbose":           -1,
}

_LGB_REG_PARAMS: dict[str, Any] = {
    "n_estimators":      500,
    "max_depth":         5,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  5,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "objective":         "regression",
    "metric":            "mae",
    "random_state":      42,
    "n_jobs":            -1,
    "verbose":           -1,
}


# ─── Public API ───────────────────────────────────────────────────────────────


def train_fold(
    X_tr: np.ndarray,
    y_cls_tr: np.ndarray,
    y_max_tr: np.ndarray,
    y_min_tr: np.ndarray,
    X_va: np.ndarray,
    y_cls_va: np.ndarray,
    y_max_va: np.ndarray,
    y_min_va: np.ndarray,
    backend: str = "xgb",
) -> tuple:
    """
    Train all three heads for a single fold.

    Regression heads (max, min) are now trained on ALL samples (not just positive-class).
    This prevents out-of-distribution predictions when the classifier fires on negative samples.

    Parameters
    ----------
    backend : "xgb" (XGBoost, default) or "lgb" (LightGBM)

    Returns
    -------
    (model_cls, model_max, model_min)
    """
    pos_count = int(np.sum(y_cls_tr == 1))
    neg_count = int(np.sum(y_cls_tr == 0))
    spw = neg_count / max(pos_count, 1)
    log.info("    cls  | train=%d (pos=%d neg=%d spw=%.2f)", len(y_cls_tr), pos_count, neg_count, spw)

    if backend == "lgb":
        # ── LightGBM ──────────────────────────────────────────────────────────
        callbacks = [lgb.early_stopping(_EARLY_STOPPING, verbose=False), lgb.log_evaluation(period=-1)]

        model_cls = lgb.LGBMClassifier(**_LGB_CLS_PARAMS, scale_pos_weight=spw)
        model_cls.fit(
            X_tr, y_cls_tr,
            eval_set=[(X_va, y_cls_va)],
            callbacks=callbacks,
        )
        log.info("    cls  | best_iteration=%d", model_cls.best_iteration_)

        log.info("    max  | train=%d  val=%d", len(X_tr), len(X_va))
        model_max = lgb.LGBMRegressor(**_LGB_REG_PARAMS)
        model_max.fit(
            X_tr, y_max_tr,
            eval_set=[(X_va, y_max_va)],
            callbacks=callbacks,
        )
        log.info("    max  | best_iteration=%d", model_max.best_iteration_)

        log.info("    min  | train=%d  val=%d", len(X_tr), len(X_va))
        model_min = lgb.LGBMRegressor(**_LGB_REG_PARAMS)
        model_min.fit(
            X_tr, y_min_tr,
            eval_set=[(X_va, y_min_va)],
            callbacks=callbacks,
        )
        log.info("    min  | best_iteration=%d", model_min.best_iteration_)

    else:
        # ── XGBoost (default) ─────────────────────────────────────────────────
        model_cls = XGBClassifier(**_CLS_PARAMS, scale_pos_weight=spw, early_stopping_rounds=_EARLY_STOPPING)
        model_cls.fit(
            X_tr, y_cls_tr,
            eval_set=[(X_va, y_cls_va)],
            verbose=_VERBOSE_EVAL,
        )
        log.info("    cls  | best_iteration=%d", model_cls.best_iteration)

        log.info("    max  | train=%d  val=%d", len(X_tr), len(X_va))
        model_max = XGBRegressor(**_REG_PARAMS, early_stopping_rounds=_EARLY_STOPPING)
        model_max.fit(
            X_tr, y_max_tr,
            eval_set=[(X_va, y_max_va)],
            verbose=_VERBOSE_EVAL,
        )
        log.info("    max  | best_iteration=%d", model_max.best_iteration)

        log.info("    min  | train=%d  val=%d", len(X_tr), len(X_va))
        model_min = XGBRegressor(**_REG_PARAMS, early_stopping_rounds=_EARLY_STOPPING)
        model_min.fit(
            X_tr, y_min_tr,
            eval_set=[(X_va, y_min_va)],
            verbose=_VERBOSE_EVAL,
        )
        log.info("    min  | best_iteration=%d", model_min.best_iteration)

    return model_cls, model_max, model_min


def fit_calibrator(
    oof_scores: np.ndarray,
    oof_labels: np.ndarray,
) -> LogisticRegression:
    """
    Fit a Platt Scaling calibrator on OOF classifier scores.

    Parameters
    ----------
    oof_scores : raw predict_proba[:, 1] from XGBClassifier across all folds.
    oof_labels : true y_class values corresponding to oof_scores.

    Returns
    -------
    Fitted LogisticRegression (1-feature, calibration only).
    """
    cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
    cal.fit(oof_scores.reshape(-1, 1), oof_labels)
    log.info("Calibrator fitted on %d OOF samples.", len(oof_labels))
    return cal


def save_models(
    model_cls,
    model_max,
    model_min,
    calibrator: LogisticRegression,
    output_dir: Path,
    backend: str = "xgb",
) -> None:
    """Save all model artifacts to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if backend == "lgb":
        model_cls.booster_.save_model(str(output_dir / "model_cls.lgb"))
        model_max.booster_.save_model(str(output_dir / "model_max.lgb"))
        model_min.booster_.save_model(str(output_dir / "model_min.lgb"))
    else:
        model_cls.save_model(output_dir / "model_cls.json")
        model_max.save_model(output_dir / "model_max.json")
        model_min.save_model(output_dir / "model_min.json")

    with open(output_dir / "calibrator.pkl", "wb") as f:
        pickle.dump(calibrator, f)

    log.info("Models saved to %s", output_dir)


def load_models(
    model_dir: Path,
) -> tuple:
    """
    Load all model artifacts from model_dir.

    Detects backend automatically: if model_cls.lgb exists → LightGBM, else XGBoost.
    Returns (model_cls, model_max, model_min, calibrator).
    For LightGBM, model_cls/max/min are lgb.Booster objects.
    """
    if (model_dir / "model_cls.lgb").exists():
        model_cls = lgb.Booster(model_file=str(model_dir / "model_cls.lgb"))
        model_max = lgb.Booster(model_file=str(model_dir / "model_max.lgb"))
        model_min = lgb.Booster(model_file=str(model_dir / "model_min.lgb"))
    else:
        model_cls = XGBClassifier()
        model_cls.load_model(model_dir / "model_cls.json")
        model_max = XGBRegressor()
        model_max.load_model(model_dir / "model_max.json")
        model_min = XGBRegressor()
        model_min.load_model(model_dir / "model_min.json")

    with open(model_dir / "calibrator.pkl", "rb") as f:
        calibrator = pickle.load(f)

    return model_cls, model_max, model_min, calibrator
