"""
Load trained models and produce a calibrated signal probability.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.train.model import load_models

_DEFAULT_MODEL_DIR = Path(__file__).parent.parent.parent / "models"


def load_signal_models(model_dir: Path = _DEFAULT_MODEL_DIR):
    """
    Load model artifacts and feature column list.

    Returns
    -------
    (model_cls, model_max, model_min, calibrator, feature_cols)
    """
    model_dir = Path(model_dir)
    model_cls, model_max, model_min, calibrator = load_models(model_dir)

    feat_file = model_dir / "feature_cols.txt"
    if not feat_file.exists():
        raise FileNotFoundError(f"feature_cols.txt not found in {model_dir}")
    feature_cols = feat_file.read_text().strip().splitlines()

    return model_cls, model_max, model_min, calibrator, feature_cols


def predict(
    feature_row: pd.DataFrame,
    model_cls,
    calibrator,
    feature_cols: list[str],
) -> float:
    """
    Produce a calibrated signal probability for a single feature row.

    Parameters
    ----------
    feature_row  : single-row DataFrame from compute_live_features()
    model_cls    : loaded XGBClassifier
    calibrator   : fitted Platt scaling LogisticRegression
    feature_cols : ordered list of 131 feature names (from feature_cols.txt)

    Returns
    -------
    Calibrated probability in [0, 1].  Returns NaN if features are missing.
    """
    missing = [c for c in feature_cols if c not in feature_row.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} feature columns: {missing[:5]} ...")

    X = feature_row[feature_cols].values.astype(np.float32)

    # NaN values are acceptable — XGBoost handles them natively via default
    # split direction.  Warn only if ALL features are NaN (likely a fetch error).
    nan_count = int(np.isnan(X).sum())
    if nan_count == X.size:
        return float("nan")

    raw_prob = model_cls.predict_proba(X)[:, 1]
    cal_prob  = calibrator.predict_proba(raw_prob.reshape(-1, 1))[:, 1]
    return float(cal_prob[0])
