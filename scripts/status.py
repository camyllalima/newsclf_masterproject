"""Show a summary of the current project state. No API calls.

Reports corpus size, results CSV stats, valid label rates, and cache entries.

Usage:
    python scripts/status.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from newsclf.config import CACHE_DIR, LABELS_CNN18, PREDICTIONS_DIR
from newsclf.corpus_loader import load_corpus

log = logging.getLogger(__name__)


def _print_csv_stats(label: str, path: Path, has_pct_cols: bool = False) -> None:
    """Print stats for a results CSV if it exists."""
    if not path.exists():
        log.info("  %s: NOT FOUND (%s)", label, path)
        return

    df = pd.read_csv(path)
    n_uids = df["uid"].nunique()
    log.info("  %s", label)
    log.info("    File:       %s", path.name)
    log.info("    Records:    %d", len(df))
    log.info("    UIDs:       %d", n_uids)

    if "input_type" in df.columns:
        log.info("    By input_type:")
        for it, count in df["input_type"].value_counts().items():
            log.info("      %-25s %d", it, count)

    if "predicted_label" in df.columns:
        valid = df["predicted_label"].notna().sum()
        log.info("    Valid labels:    %d/%d (%.1f%%)", valid, len(df), valid / len(df) * 100)

    if has_pct_cols:
        label_cols = [c for c in LABELS_CNN18 if c in df.columns]
        valid = df[label_cols].notna().all(axis=1).sum()
        log.info("    Valid dists:     %d/%d (%.1f%%)", valid, len(df), valid / len(df) * 100)


def main() -> None:
    # 1. Corpus
    log.info("=" * 60)
    log.info("CORPUS")
    log.info("=" * 60)
    articles = load_corpus()
    log.info("  Articles loaded: %d", len(articles))
    log.info("")

    # 2. Results CSVs (per-provider)
    log.info("=" * 60)
    log.info("RESULTS")
    log.info("=" * 60)
    single_files = sorted(PREDICTIONS_DIR.glob("*/results_single_*.csv"))
    percent_files = sorted(PREDICTIONS_DIR.glob("*/results_percent_*.csv"))

    if not single_files and not percent_files:
        log.info("  No result files found in %s", PREDICTIONS_DIR)
    else:
        for f in single_files:
            _print_csv_stats(f"Single ({f.stem})", f)
            log.info("")
        for f in percent_files:
            _print_csv_stats(f"Percent ({f.stem})", f, has_pct_cols=True)
            log.info("")

    # 3. Cache
    log.info("=" * 60)
    log.info("CACHE")
    log.info("=" * 60)
    if CACHE_DIR.is_dir():
        n_files = sum(1 for f in CACHE_DIR.iterdir() if f.is_file())
        log.info("  Directory: %s", CACHE_DIR)
        log.info("  Entries:   %d", n_files)
    else:
        log.info("  Cache directory not found (%s)", CACHE_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
