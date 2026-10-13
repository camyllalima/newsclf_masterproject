"""Build a concordance table from per-provider results_single CSVs.

For each article (uid), pivots the three input types into columns and adds
pairwise agreement flags. Saves to results/agreement.csv.

Usage:
    python scripts/build_agreement.py
"""

from __future__ import annotations

import logging

import pandas as pd

from newsclf.config import PREDICTIONS_DIR, RESULTS_DIR

log = logging.getLogger(__name__)

INPUT_COLS: list[tuple[str, str]] = [
    ("full_text", "label_full_text"),
    ("extractive_summary", "label_extractive_summary"),
    ("abstractive_summary", "label_abstractive_summary"),
]

PAIRS: list[tuple[str, str, str]] = [
    ("label_full_text", "label_extractive_summary", "full_vs_extractive"),
    ("label_full_text", "label_abstractive_summary", "full_vs_abstractive"),
    ("label_extractive_summary", "label_abstractive_summary", "extractive_vs_abstractive"),
]


def build_agreement_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot input types into columns and compute pairwise agreement."""
    base = df[["uid"]].drop_duplicates(subset="uid")

    for input_type, col_name in INPUT_COLS:
        subset = df.loc[df["input_type"] == input_type, ["uid", "predicted_label"]].rename(
            columns={"predicted_label": col_name}
        )
        base = base.merge(subset, on="uid", how="left")

    for col_a, col_b, flag_col in PAIRS:
        both_present = base[col_a].notna() & base[col_b].notna()
        base[flag_col] = None
        base.loc[both_present, flag_col] = (
            base.loc[both_present, col_a] == base.loc[both_present, col_b]
        ).map({True: "Yes", False: "No"})

    return base


def main() -> None:
    """Load all per-provider single CSVs, build concordance table, and print summary."""
    files = sorted(PREDICTIONS_DIR.glob("*/results_single_*.csv"))
    if not files:
        log.info("No results_single_*.csv files found in %s", PREDICTIONS_DIR)
        return

    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    log.info("Read %d records from %d files", len(df), len(files))

    agreement = build_agreement_table(df)

    out_path = RESULTS_DIR / "agreement.csv"
    agreement.to_csv(out_path, index=False)
    log.info("Table saved to %s (%d articles)", out_path, len(agreement))

    log.info("")
    log.info("=== Agreement summary ===")
    for col_a, col_b, flag_col in PAIRS:
        col = agreement[flag_col]
        valid = col.dropna()
        if valid.empty:
            log.info("  %s: no data", flag_col)
            continue
        n_sim = (valid == "Yes").sum()
        rate = n_sim / len(valid) * 100
        log.info("  %s: %d/%d (%.1f%%)", flag_col, n_sim, len(valid), rate)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
