"""Compute Cohen's Kappa, macro-F1, and label adherence from result CSVs."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.metrics import classification_report, cohen_kappa_score, f1_score

from newsclf.config import LABELS_CNN18, PREDICTIONS_DIR

log = logging.getLogger(__name__)

GROUP_KEYS = ["provider", "input_type"]


def compute_metrics(df: pd.DataFrame) -> dict[str, float] | None:
    """Return kappa and macro-F1 for a results DataFrame.

    Expects columns ``true_label`` and ``predicted_label``.
    Returns None when ``true_label`` is missing from the DataFrame.
    """
    if "true_label" not in df.columns:
        log.warning("Column 'true_label' not found — skipping metrics.")
        return None

    valid = df.dropna(subset=["true_label", "predicted_label"])
    if valid.empty:
        log.warning("No valid rows with both true and predicted labels.")
        return None

    y_true = valid["true_label"]
    y_pred = valid["predicted_label"]

    kappa = cohen_kappa_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"kappa": kappa, "f1_macro": f1_macro}


def compute_adherence() -> pd.DataFrame:
    """Compute label adherence rate per provider and input_type.

    Adherence = (valid responses / total responses) for each (provider, input_type).
    For single CSVs: valid means predicted_label is a known label in LABELS_CNN18.
    For percent CSVs: valid means all 18 label columns are non-NaN.

    Returns a DataFrame with columns: provider, input_type, metodo, total, validos, taxa.

    Note: column names (metodo, validos, taxa) are frozen CSV interface — do not rename.
    """
    valid_labels = set(LABELS_CNN18)
    rows: list[dict] = []

    # Single-label adherence.
    for f in sorted(PREDICTIONS_DIR.glob("*/results_single_*.csv")):
        df = pd.read_csv(f)
        for (provider, itype), grp in df.groupby(["provider", "input_type"]):
            total = len(grp)
            n_valid = int(grp["predicted_label"].isin(valid_labels).sum())
            rows.append(
                {
                    "provider": provider,
                    "input_type": itype,
                    "metodo": "single",
                    "total": total,
                    "validos": n_valid,
                    "taxa": n_valid / total if total else 0.0,
                }
            )

    # Percent adherence.
    label_cols = LABELS_CNN18
    for f in sorted(PREDICTIONS_DIR.glob("*/results_percent_*.csv")):
        df = pd.read_csv(f)
        present_cols = [c for c in label_cols if c in df.columns]
        for (provider, itype), grp in df.groupby(["provider", "input_type"]):
            total = len(grp)
            n_valid = int(grp[present_cols].notna().all(axis=1).sum()) if present_cols else 0
            rows.append(
                {
                    "provider": provider,
                    "input_type": itype,
                    "metodo": "percent",
                    "total": total,
                    "validos": n_valid,
                    "taxa": n_valid / total if total else 0.0,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    """Compute and print Kappa and macro-F1 per provider/input_type."""
    files = sorted(PREDICTIONS_DIR.glob("*/results_single_*.csv"))
    if not files:
        log.info("No results_single_*.csv files found.")
        return

    results = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    invalid_count = int(results["predicted_label"].isna().sum())
    if invalid_count:
        log.info("Warning: %d invalid answers ignored.\n", invalid_count)
    results = results.dropna(subset=["predicted_label"])

    for keys, group in results.groupby(GROUP_KEYS):
        provider, input_type = keys
        log.info("=== %s | %s ===", provider, input_type)
        log.info("  Records: %d", len(group))

        metrics = compute_metrics(group)
        if metrics is not None:
            log.info("  Kappa:     %.4f", metrics["kappa"])
            log.info("  F1-macro:  %.4f", metrics["f1_macro"])
            log.info(
                "\n%s",
                classification_report(
                    group["true_label"], group["predicted_label"], zero_division=0
                ),
            )
        log.info("")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
