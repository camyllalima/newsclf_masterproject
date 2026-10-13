"""Reusable analysis functions for notebooks and scripts."""

from __future__ import annotations

import pandas as pd

from newsclf.config import LABELS_CNN18, PREDICTIONS_DIR


def load_result_csvs() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Return (df_single, df_percent) or None for each missing set."""
    single_files = sorted(PREDICTIONS_DIR.glob("*/results_single_*.csv"))
    percent_files = sorted(PREDICTIONS_DIR.glob("*/results_percent_*.csv"))
    df_s = (
        pd.concat([pd.read_csv(f) for f in single_files], ignore_index=True)
        if single_files
        else None
    )
    df_p = (
        pd.concat([pd.read_csv(f) for f in percent_files], ignore_index=True)
        if percent_files
        else None
    )
    return df_s, df_p


def compute_top1_labels(df_percent: pd.DataFrame) -> pd.DataFrame:
    """Add top1_label, top1_tied, and top1_valid columns to percent data.

    - top1_label: the category with the highest percentage.
    - top1_tied: pipe-separated string of all categories tied at the max
      (equals top1_label when there is no tie).
    - top1_valid: False when 6+ categories are tied at the max (uniform
      response, treated as non-response); True otherwise.
    """
    label_cols = [c for c in LABELS_CNN18 if c in df_percent.columns]
    df = df_percent[df_percent[label_cols].notna().all(axis=1)].copy()

    vals = df[label_cols].astype(float)
    max_vals = vals.max(axis=1)

    # For each row, find all columns that equal the max
    tied = vals.eq(max_vals, axis=0)
    n_tied = tied.sum(axis=1)

    df["top1_label"] = vals.idxmax(axis=1)
    df["top1_tied"] = tied.apply(
        lambda row: "|".join(sorted(col for col, is_max in row.items() if is_max)),
        axis=1,
    )
    df["top1_valid"] = n_tied < 6

    return df
