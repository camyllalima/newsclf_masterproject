"""Evaluate LLM predictions against the weighted human ground-truth.

Builds the answer key (if not cached), then computes per-article
and aggregated accuracy metrics for each provider.

Usage:
    python scripts/evaluate_accuracy.py                     # all providers
    python scripts/evaluate_accuracy.py --provider chatgpt  # ChatGPT only
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from newsclf.answer_key import build_answer_key, save_answer_key
from newsclf.config import (
    ANSWER_KEY_DIR,
    EVALUATION_DIR,
    LABELS_CNN18,
    MODELS,
    PREDICTIONS_DIR,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------


def _load_answer_key() -> pd.DataFrame:
    """Load or build the answer_key CSV."""
    cached = ANSWER_KEY_DIR / "answer_key.csv"
    if cached.exists():
        df = pd.read_csv(cached)
        log.info("Answer key loaded from %s (%d articles)", cached, len(df))
        return df
    log.info("Answer key not found — building...")
    df = build_answer_key()
    save_answer_key(df)
    return df


def _load_predictions(provider: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Return (df_single, df_percent) for a provider, or None if missing."""
    single_path = PREDICTIONS_DIR / provider / f"results_single_{provider}.csv"
    percent_path = PREDICTIONS_DIR / provider / f"results_percent_{provider}.csv"
    df_s = pd.read_csv(single_path) if single_path.exists() else None
    df_p = pd.read_csv(percent_path) if percent_path.exists() else None
    return df_s, df_p


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def _percent_top1(row: pd.Series, label_cols: list[str]) -> str | None:
    """Return the top-1 label from percent predictions."""
    vals = row[label_cols].astype(float)
    if vals.isna().all():
        return None
    return str(vals.idxmax())


def _percent_top3(row: pd.Series, label_cols: list[str]) -> list[str]:
    """Return the top-3 labels from percent predictions."""
    vals = row[label_cols].astype(float).dropna()
    return list(vals.nlargest(3).index)


def _overlap_count(pred_top3: list[str], answer_key_labels: list[str]) -> int:
    """Count how many of pred_top3 appear in answer_key_labels."""
    gab_set = set(answer_key_labels)
    return sum(1 for lb in pred_top3 if lb in gab_set)


def evaluate_provider(
    provider: str,
    df_answer_key: pd.DataFrame,
) -> pd.DataFrame | None:
    """Evaluate a single provider and return per-article results."""
    df_single, df_percent = _load_predictions(provider)

    if df_single is None and df_percent is None:
        log.warning("[%s] No results found — skipping.", provider)
        return None

    # Parse answer_key into usable structures.
    gab_map: dict[str, dict] = {}
    for _, row in df_answer_key.iterrows():
        ranking = json.loads(row["ranking_json"])
        top1 = set(row["gold_label"].split("|")) if row["gold_label"] else set()
        top3_gab = list(ranking.keys())[:3]
        all_labels = list(ranking.keys())
        gab_map[row["uid"]] = {
            "top1": top1,
            "top3": top3_gab,
            "all": all_labels,
        }

    label_cols = [c for c in LABELS_CNN18 if df_percent is not None and c in df_percent.columns]

    # Determine input_types from available data.
    if df_single is not None:
        input_types = sorted(df_single["input_type"].unique())
    else:
        input_types = sorted(df_percent["input_type"].unique())

    results: list[dict] = []

    for input_type in input_types:
        # Single predictions for this input_type.
        single_map: dict[str, str] = {}
        if df_single is not None:
            sub = df_single[df_single["input_type"] == input_type]
            for _, r in sub.iterrows():
                if pd.notna(r["predicted_label"]):
                    single_map[r["uid"]] = r["predicted_label"]

        # Percent predictions for this input_type.
        pct_rows: dict[str, pd.Series] = {}
        if df_percent is not None:
            sub = df_percent[df_percent["input_type"] == input_type]
            for _, r in sub.iterrows():
                pct_rows[r["uid"]] = r

        # Combine UIDs.
        all_uids = set(single_map.keys()) | set(pct_rows.keys())

        for uid in sorted(all_uids):
            gab = gab_map.get(uid)
            if gab is None:
                continue

            pred_single = single_map.get(uid)
            hit_single = pred_single in gab["top1"] if pred_single else None

            pred_pct_top1 = None
            hit_pct_top1 = None
            overlap_all = None
            overlap_top3 = None

            if uid in pct_rows and label_cols:
                pred_pct_top1 = _percent_top1(pct_rows[uid], label_cols)
                if pred_pct_top1:
                    hit_pct_top1 = pred_pct_top1 in gab["top1"]
                    top3_pct = _percent_top3(pct_rows[uid], label_cols)
                    overlap_all = _overlap_count(top3_pct, gab["all"])
                    overlap_top3 = _overlap_count(top3_pct, gab["top3"])

            results.append(
                {
                    "uid": uid,
                    "provider": provider,
                    "input_type": input_type,
                    "pred_single": pred_single,
                    "hit_single": hit_single,
                    "pred_percent_top1": pred_pct_top1,
                    "hit_percent_top1": hit_pct_top1,
                    "overlap_vs_all": overlap_all,
                    "overlap_vs_top3": overlap_top3,
                }
            )

    if not results:
        return None

    return pd.DataFrame(results)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-article results into a summary by (provider, input_type)."""
    rows: list[dict] = []
    for (provider, input_type), grp in df.groupby(["provider", "input_type"]):
        row: dict = {"provider": provider, "input_type": input_type}

        # Single accuracy.
        valid_s = grp["hit_single"].dropna()
        row["n_single"] = len(valid_s)
        row["accuracy_single"] = valid_s.mean() if len(valid_s) > 0 else None

        # Percent top-1 accuracy.
        valid_p = grp["hit_percent_top1"].dropna()
        row["n_percent"] = len(valid_p)
        row["accuracy_percent_top1"] = valid_p.mean() if len(valid_p) > 0 else None

        # Overlap stats.
        for col in ("overlap_vs_all", "overlap_vs_top3"):
            vals = grp[col].dropna()
            row[f"{col}_media"] = vals.mean() if len(vals) > 0 else None
            for k in range(4):
                row[f"{col}_{k}"] = int((vals == k).sum()) if len(vals) > 0 else 0

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LLM accuracy vs ground-truth.")
    parser.add_argument(
        "--provider",
        choices=list(MODELS.keys()),
        default=None,
        help="Evaluate only this provider.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df_answer_key = _load_answer_key()

    # Report tie stats.
    n_ties = int(df_answer_key["has_tie"].sum())
    log.info(
        "Answer key: %d articles, %d with top-1 tie (%.1f%%)",
        len(df_answer_key),
        n_ties,
        100 * n_ties / len(df_answer_key),
    )

    if args.provider:
        providers = [args.provider]
    else:
        providers = list(MODELS.keys())

    all_details: list[pd.DataFrame] = []
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    for provider in providers:
        df_detail = evaluate_provider(provider, df_answer_key)
        if df_detail is None:
            continue

        # Save per-provider detail.
        detail_path = EVALUATION_DIR / f"accuracy_{provider}.csv"
        df_detail.to_csv(detail_path, index=False)
        log.info("Details saved to %s (%d rows)", detail_path, len(df_detail))
        all_details.append(df_detail)

    if not all_details:
        log.warning("No results to evaluate.")
        return

    df_all = pd.concat(all_details, ignore_index=True)
    summary = build_summary(df_all)

    summary_path = EVALUATION_DIR / "accuracy_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info("Summary saved to %s\n", summary_path)

    # Print summary table.
    log.info("=" * 75)
    log.info("ACCURACY SUMMARY")
    log.info("=" * 75)
    for _, row in summary.iterrows():
        log.info("")
        log.info(
            "  [%s] %s",
            row["provider"],
            row["input_type"],
        )
        if pd.notna(row.get("accuracy_single")):
            log.info(
                "    Single:       %.1f%% (%d samples)",
                row["accuracy_single"] * 100,
                row["n_single"],
            )
        if pd.notna(row.get("accuracy_percent_top1")):
            log.info(
                "    Percent top1: %.1f%% (%d samples)",
                row["accuracy_percent_top1"] * 100,
                row["n_percent"],
            )
        if pd.notna(row.get("overlap_vs_all_media")):
            log.info(
                "    Overlap vs all (mean): %.2f/3",
                row["overlap_vs_all_media"],
            )
            log.info(
                "      distribution: 0=%d  1=%d  2=%d  3=%d",
                row["overlap_vs_all_0"],
                row["overlap_vs_all_1"],
                row["overlap_vs_all_2"],
                row["overlap_vs_all_3"],
            )
        if pd.notna(row.get("overlap_vs_top3_media")):
            log.info(
                "    Overlap vs top3 (mean):  %.2f/3",
                row["overlap_vs_top3_media"],
            )
            log.info(
                "      distribution: 0=%d  1=%d  2=%d  3=%d",
                row["overlap_vs_top3_0"],
                row["overlap_vs_top3_1"],
                row["overlap_vs_top3_2"],
                row["overlap_vs_top3_3"],
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
