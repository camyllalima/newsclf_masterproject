"""Compute the human accuracy baseline via leave-one-out.

For each annotator, builds a answer_key from the OTHER two annotators and
checks if the annotator's Class1 (top pick) matches the leave-one-out top-1.
Also reports accuracy against the full (3-annotator) answer_key for comparison.

Finally, places the LLM (ChatGPT single) accuracy side by side — evaluated
against the full 3-annotator answer_key (the fair analogue, since the LLM is
not one of the annotators).

Usage:
    python scripts/human_baseline.py
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from newsclf.answer_key import (
    ANNOTATOR_NAMES,
    SHEET,
    XLSX_PATH,
    annotator_class1,
    compute_ranking,
    top1_from_ranking,
)
from newsclf.config import EVALUATION_DIR, PREDICTIONS_DIR
from newsclf.corpus_loader import load_corpus

log = logging.getLogger(__name__)


def _load_xlsx() -> tuple[pd.DataFrame, dict[str, str]]:
    """Load the spreadsheet and build title→uid mapping."""
    from newsclf.answer_key import _normalise_title

    df = pd.read_excel(XLSX_PATH, sheet_name=SHEET)
    articles = load_corpus()
    title_to_uid = {_normalise_title(a.title): a.uid for a in articles}
    df["uid"] = df["News"].astype(str).apply(
        lambda t: title_to_uid.get(_normalise_title(t))
    )
    df = df[df["uid"].notna()].copy()
    return df, title_to_uid


def compute_human_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute leave-one-out and full-answer_key accuracy for each annotator."""
    results: list[dict] = []

    for ann in ANNOTATOR_NAMES:
        others = [a for a in ANNOTATOR_NAMES if a != ann]
        n_total = 0
        hits_loo = 0
        hits_full = 0

        for _, row in df.iterrows():
            class1 = annotator_class1(row, ann)
            if class1 is None:
                continue
            n_total += 1

            # Leave-one-out: answer_key from the other 2.
            ranking_loo = compute_ranking(row, annotators=others)
            top1_loo, _ = top1_from_ranking(ranking_loo)
            if class1 in top1_loo:
                hits_loo += 1

            # Full answer_key (all 3 — includes self).
            ranking_full = compute_ranking(row)
            top1_full, _ = top1_from_ranking(ranking_full)
            if class1 in top1_full:
                hits_full += 1

        results.append(
            {
                "evaluee": ann,
                "answer_key": "other 2 (leave-one-out)",
                "accuracy": hits_loo / n_total if n_total else 0,
                "n": n_total,
            }
        )
        results.append(
            {
                "evaluee": ann,
                "answer_key": "all 3 (with contamination)",
                "accuracy": hits_full / n_total if n_total else 0,
                "n": n_total,
            }
        )

    return pd.DataFrame(results)


def compute_llm_accuracy(df_answer_key_full: pd.DataFrame) -> list[dict]:
    """Compute ChatGPT single accuracy against the full 3-annotator answer_key."""
    # Parse full answer_key.
    gab_map: dict[str, set[str]] = {}
    for _, row in df_answer_key_full.iterrows():
        ranking = json.loads(row["ranking_json"])
        top1, _ = top1_from_ranking(ranking)
        gab_map[row["uid"]] = set(top1)

    single_path = PREDICTIONS_DIR / "chatgpt" / "results_single_chatgpt.csv"
    if not single_path.exists():
        log.warning("ChatGPT single predictions not found: %s", single_path)
        return []

    df_pred = pd.read_csv(single_path)
    results: list[dict] = []

    for input_type in sorted(df_pred["input_type"].unique()):
        sub = df_pred[df_pred["input_type"] == input_type]
        hits = 0
        n = 0
        for _, row in sub.iterrows():
            pred = row.get("predicted_label")
            if pd.isna(pred):
                continue
            top1 = gab_map.get(row["uid"])
            if top1 is None:
                continue
            n += 1
            if pred in top1:
                hits += 1

        label_map = {
            "full_text": "ChatGPT single (full text)",
            "extractive_summary": "ChatGPT single (extractive summary)",
            "abstractive_summary": "ChatGPT single (abstractive summary)",
        }
        results.append(
            {
                "evaluee": label_map.get(input_type, input_type),
                "answer_key": "all 3 (full answer key)",
                "accuracy": hits / n if n else 0,
                "n": n,
            }
        )

    return results


def main() -> None:
    df_xlsx, _ = _load_xlsx()
    log.info("Spreadsheet loaded: %d articles\n", len(df_xlsx))

    # --- Human accuracy ---
    df_human = compute_human_accuracy(df_xlsx)

    # Compute human average (leave-one-out only).
    loo_rows = df_human[df_human["answer_key"].str.contains("leave-one-out")]
    avg_loo = loo_rows["accuracy"].mean()

    # --- LLM accuracy ---
    from newsclf.answer_key import build_answer_key
    from newsclf.config import ANSWER_KEY_DIR

    cached = ANSWER_KEY_DIR / "answer_key.csv"
    if cached.exists():
        df_answer_key = pd.read_csv(cached)
    else:
        df_answer_key = build_answer_key()

    llm_results = compute_llm_accuracy(df_answer_key)

    # --- Build final table ---
    rows: list[dict] = []

    # Human leave-one-out rows.
    for _, r in loo_rows.iterrows():
        rows.append(dict(r))

    rows.append(
        {
            "evaluee": "Human mean (leave-one-out)",
            "answer_key": "other 2 (leave-one-out)",
            "accuracy": avg_loo,
            "n": int(loo_rows["n"].mean()),
        }
    )

    for r in llm_results:
        rows.append(r)

    df_final = pd.DataFrame(rows)

    # --- Save ---
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVALUATION_DIR / "human_baseline.csv"
    df_final.to_csv(out_path, index=False)
    log.info("Table saved to %s\n", out_path)

    # --- Print ---
    log.info("=" * 75)
    log.info("HUMAN BASELINE — Comparative accuracy")
    log.info("=" * 75)
    log.info("")
    log.info(
        "  %-42s  %-32s  %8s  %5s",
        "Evaluee", "Answer key", "Accuracy", "n",
    )
    log.info("  " + "-" * 71)
    for _, r in df_final.iterrows():
        log.info(
            "  %-42s  %-32s  %7.1f%%  %5d",
            r["evaluee"],
            r["answer_key"],
            r["accuracy"] * 100,
            r["n"],
        )

    # --- Also show contamination comparison ---
    log.info("")
    log.info("=" * 75)
    log.info("CONTEXT — Contamination effect (human vs answer key with/without self)")
    log.info("=" * 75)
    log.info("")
    for ann in ANNOTATOR_NAMES:
        sub = df_human[df_human["evaluee"] == ann]
        loo = sub[sub["answer_key"].str.contains("leave")]["accuracy"].iloc[0]
        full = sub[sub["answer_key"].str.contains("contam")]["accuracy"].iloc[0]
        log.info(
            "  %-10s  leave-one-out: %5.1f%%   with contamination: %5.1f%%   delta: %+.1f pp",
            ann, loo * 100, full * 100, (full - loo) * 100,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
