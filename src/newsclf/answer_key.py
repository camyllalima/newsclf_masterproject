"""Build weighted ground-truth (ClassesHumansX) from the annotation spreadsheet."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from newsclf.config import ANSWER_KEY_DIR, DB3_DIR, LABELS_CNN18
from newsclf.corpus_loader import load_corpus

log = logging.getLogger(__name__)

XLSX_PATH = DB3_DIR / "classes.xlsx"
SHEET = "alldiv"

# Human annotator columns grouped by annotator, ordered by position (1st, 2nd, 3rd).
ANNOTATOR_NAMES = ["Human1", "Hum2", "Hum3"]

ANNOTATOR_COLS: dict[str, list[str]] = {
    "Human1": ["Human1Class1", "Human1Class2", "Human1Class3"],
    "Hum2": ["Hum2Class1", "Hum2Class2", "Hum2Class3"],
    "Hum3": ["Hum3Class1", "Hum3Class2", "Hum3Class3"],
}

_ALL_ANNOTATOR_COLS: list[list[str]] = list(ANNOTATOR_COLS.values())

# Weights by position: 1st = 3, 2nd = 2, 3rd = 1.
_POSITION_WEIGHTS = [3, 2, 1]

# Case-insensitive lookup: lowered label → canonical LABELS_CNN18 form.
_CANONICAL: dict[str, str] = {lb.lower(): lb for lb in LABELS_CNN18}


def _normalise_title(raw: str) -> str:
    """Normalise a spreadsheet title for matching against corpus titles.

    Re-uses the same logic as check_ground_truth.normalise_title, inlined
    to avoid importing from scripts/.
    """
    import html as _html
    import re
    import unicodedata

    text = _html.unescape(raw)
    text = text.replace("&apost;", "'").replace("&quot;", '"')
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"[\"''\u201c\u201d]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(".,;:!?- ")
    return text


def canonicalise_label(raw: str) -> str | None:
    """Map a raw label string to its canonical LABELS_CNN18 form, or None."""
    return _CANONICAL.get(raw.strip().lower())


def compute_ranking(
    row: pd.Series,
    annotators: list[str] | None = None,
) -> dict[str, int]:
    """Compute the weighted class ranking for a single article row.

    Args:
        row: A row from the classes.xlsx DataFrame.
        annotators: Which annotators to include. None = all three.

    Returns {canonical_label: total_points} sorted descending by points.
    """
    if annotators is None:
        cols_list = _ALL_ANNOTATOR_COLS
    else:
        cols_list = [ANNOTATOR_COLS[a] for a in annotators]

    scores: dict[str, int] = {}
    for ann_cols in cols_list:
        for pos, col in enumerate(ann_cols):
            raw = row.get(col)
            if pd.isna(raw) or not str(raw).strip():
                continue
            label = canonicalise_label(str(raw))
            if label is None:
                log.warning("Unknown label ignored: %r", raw)
                continue
            scores[label] = scores.get(label, 0) + _POSITION_WEIGHTS[pos]
    # Sort descending by score.
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


def annotator_class1(row: pd.Series, annotator: str) -> str | None:
    """Return the canonical Class1 (top pick) of a single annotator, or None."""
    col = ANNOTATOR_COLS[annotator][0]
    raw = row.get(col)
    if pd.isna(raw) or not str(raw).strip():
        return None
    return canonicalise_label(str(raw))


def top1_from_ranking(ranking: dict[str, int]) -> tuple[list[str], bool]:
    """Return (list of top-1 labels, has_tie).

    If multiple labels share the highest score, all are returned.
    """
    if not ranking:
        return [], False
    max_score = max(ranking.values())
    top = [lb for lb, sc in ranking.items() if sc == max_score]
    return top, len(top) > 1


def build_answer_key(
    xlsx_path: Path = XLSX_PATH,
    sheet: str = SHEET,
) -> pd.DataFrame:
    """Build the full ClassesHumansX dataframe.

    Returns a DataFrame with columns:
        uid, ranking_json, gold_label, has_tie
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    articles = load_corpus()
    title_to_uid = {_normalise_title(a.title): a.uid for a in articles}

    rows: list[dict] = []
    for _, row in df.iterrows():
        title_norm = _normalise_title(str(row["News"]))
        uid = title_to_uid.get(title_norm)
        if uid is None:
            log.warning("Title not matched in corpus: %s", row["News"])
            continue

        ranking = compute_ranking(row)
        top1, has_tie = top1_from_ranking(ranking)

        rows.append(
            {
                "uid": uid,
                "ranking_json": json.dumps(ranking, ensure_ascii=False),
                "gold_label": "|".join(top1),
                "has_tie": has_tie,
            }
        )

    return pd.DataFrame(rows)


def save_answer_key(df: pd.DataFrame, out_dir: Path = ANSWER_KEY_DIR) -> Path:
    """Save the answer key CSV and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "answer_key.csv"
    df.to_csv(path, index=False)
    log.info("Answer key saved to %s (%d articles)", path, len(df))
    return path
