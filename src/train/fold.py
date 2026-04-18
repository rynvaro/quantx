"""
Walk-forward fold generator for time-series cross-validation.

Generates (train_indices, val_indices) pairs where:
- Each fold's validation window is a fixed number of months
- A gap of N days separates train from val to prevent indirect leakage
- Folds walk forward in time; train set grows with each fold

Usage
-----
    from src.train.fold import walk_forward_folds
    for fold_num, (tr_idx, va_idx) in enumerate(walk_forward_folds(df)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]
"""

import logging
from typing import Iterator

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def walk_forward_folds(
    df: pd.DataFrame,
    n_folds: int = 6,
    val_months: int = 4,
    gap_days: int = 5,
    min_train_months: int = 6,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Generate walk-forward (train, val) index pairs.

    The full timeline is divided so that the last `n_folds * val_months`
    months form the rolling validation windows, each separated from the
    preceding training data by `gap_days`.

    Parameters
    ----------
    df              : DataFrame with a UTC DatetimeIndex (5m sample base).
    n_folds         : Number of folds to generate.
    val_months      : Length of each validation window in months.
    gap_days        : Number of days to exclude between train end and val start.
    min_train_months: Minimum training length required; folds that don't meet
                      this are skipped with a warning.

    Yields
    ------
    (train_idx, val_idx) : numpy arrays of integer positional indices.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    total_start = df.index.min()
    total_end   = df.index.max()

    # The validation windows start at total_end - n_folds * val_months
    val_total_delta = pd.DateOffset(months=n_folds * val_months)
    val_zone_start  = total_end - val_total_delta

    log.info(
        "Walk-forward split: %d folds × %d months val | gap=%d days",
        n_folds, val_months, gap_days,
    )
    log.info("  Full data   : %s → %s", total_start.date(), total_end.date())
    log.info("  Val zone    : %s → %s", val_zone_start.date(), total_end.date())

    gap = pd.Timedelta(days=gap_days)

    for fold in range(n_folds):
        val_start = val_zone_start + pd.DateOffset(months=fold * val_months)
        val_end   = val_zone_start + pd.DateOffset(months=(fold + 1) * val_months)

        train_end = val_start - gap

        # Check minimum training length
        train_months = (train_end - total_start).days / 30.44
        if train_months < min_train_months:
            log.warning(
                "  Fold %d skipped: only %.1f months of training data (min=%d)",
                fold + 1, train_months, min_train_months,
            )
            continue

        train_mask = (df.index < train_end)
        val_mask   = (df.index >= val_start) & (df.index < val_end)

        train_idx = np.where(train_mask)[0]
        val_idx   = np.where(val_mask)[0]

        if len(train_idx) == 0 or len(val_idx) == 0:
            log.warning("  Fold %d skipped: empty train or val split.", fold + 1)
            continue

        log.info(
            "  Fold %d | train %s→%s (%d rows) | val %s→%s (%d rows)",
            fold + 1,
            df.index[train_idx[0]].date(),
            df.index[train_idx[-1]].date(),
            len(train_idx),
            df.index[val_idx[0]].date(),
            df.index[val_idx[-1]].date(),
            len(val_idx),
        )

        yield train_idx, val_idx
