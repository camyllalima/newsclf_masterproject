"""Count sentences per article for each input type and report statistics.

Uses a regex-based sentence splitter that masks common abbreviations before
splitting on sentence-ending punctuation (.!?) followed by an uppercase start.

Usage:
    python scripts/measure_sentences.py
"""

from __future__ import annotations

import logging

import numpy as np

from newsclf.corpus_loader import load_corpus
from newsclf.text_utils import count_sentences

log = logging.getLogger(__name__)


def main() -> None:
    articles = load_corpus()
    log.info("Corpus: %d articles\n", len(articles))

    fields = [
        ("full_text", "Full text"),
        ("gold_standard", "Extractive summary"),
        ("highlights", "Abstractive summary"),
    ]

    for attr, label in fields:
        counts = np.array([count_sentences(getattr(a, attr)) for a in articles])

        log.info("--- %s ---", label)
        log.info("  Mean:          %6.1f", counts.mean())
        log.info("  Median:        %6.1f", np.median(counts))
        log.info("  Min:           %6d", counts.min())
        log.info("  Max:           %6d", counts.max())
        log.info("  Std dev:       %6.1f", counts.std())
        log.info("")

    # Distribution for gold_standard and highlights.
    for attr, label in fields[1:]:
        counts = np.array([count_sentences(getattr(a, attr)) for a in articles])
        log.info("--- Distribution: %s ---", label)
        for k in range(1, 6):
            n = int((counts == k).sum())
            log.info("  %d sentence(s): %4d articles (%5.1f%%)", k, n, 100 * n / len(counts))
        n_5plus = int((counts >= 5).sum())
        log.info("  5+ sentences:  %4d articles (%5.1f%%)", n_5plus, 100 * n_5plus / len(counts))
        log.info("")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
