"""Basic corpus validation, run before spending API budget."""

from __future__ import annotations

import logging

from newsclf.corpus_loader import Article

log = logging.getLogger(__name__)


def validate_corpus(articles: list[Article]) -> None:
    """Log warnings about data quality issues. Raise if the list is empty."""
    if not articles:
        raise ValueError("Empty corpus — no articles loaded.")

    log.info("=== Corpus validation (%d articles) ===", len(articles))

    # 1. full_text empty or very short
    short = [(a, len(a.full_text)) for a in articles if len(a.full_text) < 100]
    if short:
        log.warning("full_text empty or < 100 chars: %d", len(short))
        for a, n in short[:10]:
            log.warning("  uid=%s title=%s (%d chars)", a.uid, a.title[:60], n)
    else:
        log.info("full_text: all >= 100 chars.")

    # 2. highlights empty
    empty_hl = [a for a in articles if not a.highlights]
    if empty_hl:
        log.warning("highlights empty: %d", len(empty_hl))
        for a in empty_hl[:10]:
            log.warning("  uid=%s title=%s", a.uid, a.title[:60])
    else:
        log.info("highlights: none empty.")

    # 3. gold_standard empty
    empty_gs = [a for a in articles if not a.gold_standard]
    if empty_gs:
        log.warning("gold_standard empty: %d", len(empty_gs))
        for a in empty_gs[:10]:
            log.warning("  uid=%s title=%s", a.uid, a.title[:60])
    else:
        log.info("gold_standard: none empty.")
