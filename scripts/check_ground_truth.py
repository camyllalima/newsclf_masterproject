"""Check alignment between the annotation spreadsheet and the db3 corpus.

Reads data/db3/classes.xlsx (sheet 'alldiv') and the corpus, normalises
titles from both sides, and reports matching/missing/duplicate entries.
Also checks that the class labels in the spreadsheet match LABELS_CNN18.

Usage:
    python scripts/check_ground_truth.py
"""

from __future__ import annotations

import html
import logging
import re
import unicodedata

import pandas as pd

from newsclf.config import LABELS_CNN18, PROJECT_ROOT
from newsclf.corpus_loader import Article, load_corpus

log = logging.getLogger(__name__)

XLSX_PATH = PROJECT_ROOT / "data" / "db3" / "classes.xlsx"
SHEET = "alldiv"

# Class columns in the spreadsheet.
CLASS_COLS = [
    "AutoClass1",
    "AutoClass2",
    "AutoClass3",
    "Human1Class1",
    "Human1Class2",
    "Human1Class3",
    "Hum2Class1",
    "Hum2Class2",
    "Hum2Class3",
    "Hum3Class1",
    "Hum3Class2",
    "Hum3Class3",
]


def normalise_title(raw: str) -> str:
    """Aggressively normalise a title for matching.

    Steps:
    1. Decode HTML entities (&apost;, &quot;, &#39;, etc.)
    2. Unicode NFKC normalisation
    3. Lowercase
    4. Replace residual XML entities (same as corpus_loader.clean_text)
    5. Strip quotes and punctuation from edges
    6. Collapse whitespace
    """
    text = html.unescape(raw)
    text = text.replace("&apost;", "'").replace("&quot;", '"')
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"[\"''\u201c\u201d]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(".,;:!?- ")
    return text


def _report_duplicates(df_xlsx: pd.DataFrame, articles: list[Article]) -> None:
    """Report duplicate titles in both the spreadsheet and the corpus."""
    log.info("=" * 60)
    log.info("DUPLICATES")
    log.info("=" * 60)

    xlsx_dup = df_xlsx[df_xlsx["title_norm"].duplicated(keep=False)]
    if not xlsx_dup.empty:
        dup_titles = sorted(xlsx_dup["title_norm"].unique())
        log.info("Spreadsheet: %d duplicate titles", len(dup_titles))
        for t in dup_titles[:10]:
            log.info("  - %s", t)
    else:
        log.info("Spreadsheet: no duplicates")

    corpus_norms = [normalise_title(a.title) for a in articles]
    seen: dict[str, int] = {}
    corpus_dups: list[str] = []
    for t in corpus_norms:
        seen[t] = seen.get(t, 0) + 1
    for t, c in seen.items():
        if c > 1:
            corpus_dups.append(t)
    if corpus_dups:
        log.info("Corpus: %d duplicate titles", len(corpus_dups))
        for t in sorted(corpus_dups)[:10]:
            log.info("  - %s", t)
    else:
        log.info("Corpus: no duplicates")


def _report_title_matching(df_xlsx: pd.DataFrame, corpus_titles: dict[str, Article]) -> None:
    """Report how many titles match between spreadsheet and corpus."""
    log.info("")
    log.info("=" * 60)
    log.info("TITLE CROSS-REFERENCE (NORMALISED)")
    log.info("=" * 60)

    xlsx_set = set(df_xlsx["title_norm"])
    corpus_set = set(corpus_titles.keys())

    matched = xlsx_set & corpus_set
    only_xlsx = xlsx_set - corpus_set
    only_corpus = corpus_set - xlsx_set

    log.info("Matched:                       %d / %d", len(matched), len(corpus_titles))
    log.info("Only in spreadsheet:           %d", len(only_xlsx))
    log.info("Only in corpus (no answer key):  %d", len(only_corpus))

    if only_xlsx:
        log.info("\n--- Titles ONLY IN SPREADSHEET (up to 20) ---")
        for t in sorted(only_xlsx)[:20]:
            log.info("  [XLSX] %s", t)

    if only_corpus:
        log.info("\n--- Titles ONLY IN CORPUS (up to 20) ---")
        for t in sorted(only_corpus)[:20]:
            log.info("  [DB3]  %s", t)


def _report_label_reconciliation(df_xlsx: pd.DataFrame) -> None:
    """Check that class labels in the spreadsheet match LABELS_CNN18."""
    log.info("")
    log.info("=" * 60)
    log.info("LABEL RECONCILIATION")
    log.info("=" * 60)

    xlsx_labels: set[str] = set()
    for col in CLASS_COLS:
        if col in df_xlsx.columns:
            vals = df_xlsx[col].dropna().astype(str).str.strip().unique()
            xlsx_labels.update(vals)

    # Normalise to title case for comparison.
    xlsx_labels_title = {lb.strip().title() for lb in xlsx_labels}
    labels_set = set(LABELS_CNN18)

    in_both = xlsx_labels_title & labels_set
    only_in_xlsx_labels = xlsx_labels_title - labels_set
    only_in_config = labels_set - xlsx_labels_title

    log.info("Labels in answer key (title case): %d", len(xlsx_labels_title))
    log.info("Labels in LABELS_CNN18:          %d", len(labels_set))
    log.info("In common:                       %d", len(in_both))

    if only_in_xlsx_labels:
        log.info("\nOnly in answer key (not in LABELS_CNN18):")
        for lb in sorted(only_in_xlsx_labels):
            # Show original casing from spreadsheet.
            originals = [x for x in xlsx_labels if x.strip().title() == lb]
            log.info("  - '%s' (original: %s)", lb, originals)

    if only_in_config:
        log.info("\nOnly in LABELS_CNN18 (not in answer key):")
        for lb in sorted(only_in_config):
            log.info("  - '%s'", lb)

    if not only_in_xlsx_labels and not only_in_config:
        log.info("\nAll 18 classes match perfectly!")


def main() -> None:
    # ── Load spreadsheet ──────────────────────────────────────────────
    df_xlsx = pd.read_excel(XLSX_PATH, sheet_name=SHEET)
    log.info("Spreadsheet: %d rows, columns: %s", len(df_xlsx), list(df_xlsx.columns))

    df_xlsx["title_norm"] = df_xlsx["News"].astype(str).apply(normalise_title)

    # ── Load XML corpus ───────────────────────────────────────────────
    articles = load_corpus()
    log.info("Corpus: %d articles\n", len(articles))

    corpus_titles = {normalise_title(a.title): a for a in articles}

    _report_duplicates(df_xlsx, articles)
    _report_title_matching(df_xlsx, corpus_titles)
    _report_label_reconciliation(df_xlsx)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
