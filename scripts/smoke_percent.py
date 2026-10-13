"""Smoke test for the percentage-distribution prompt with real API calls.

Tests with a synthetic sports text (obvious) and a real ambiguous article from
the corpus, printing the raw response and parsed distribution for each.

Usage:
    python scripts/smoke_percent.py
"""

from __future__ import annotations

import logging

from newsclf.config import LABELS_CNN18, MODELS
from newsclf.corpus_loader import load_corpus
from newsclf.llm_clients import classify
from newsclf.prompt import build_prompt_percent, extract_percent

log = logging.getLogger(__name__)

SAMPLE_TEXT = (
    "Manchester City defeated Arsenal 3-1 in a thrilling Premier League match. "
    "Haaland scored twice in the first half, sealing the victory for the home side. "
    "Arsenal's manager said the team needs to improve its defensive strategy."
)


def _classify_and_show_distribution(label: str, text: str, provider: str, model: str) -> None:
    """Build percent prompt, call the model, and print results."""
    log.info("=== %s ===", label)
    prompt = build_prompt_percent(text, LABELS_CNN18)
    raw = classify(provider, model, prompt)
    log.info("Resposta CRUA:\n%s\n", raw)

    result = extract_percent(raw, LABELS_CNN18)
    if result is not None:
        log.info("extract_percent OK — %d classes:", len(result))
        for lb, pct in sorted(result.items(), key=lambda x: -x[1]):
            bar = "#" * int(pct / 2)
            log.info("  %-25s %6.1f%%  %s", lb, pct, bar)
        log.info("  Soma: %.1f%%", sum(result.values()))
    else:
        log.warning("extract_percent retornou None — resposta malformada.")
    log.info("")


def main() -> None:
    provider, model = next(iter(MODELS.items()))
    log.info("Provider: %s (%s)\n", provider, model)

    # 1. Texto sintético óbvio (esportes).
    _classify_and_show_distribution("Texto sintetico (esportes)", SAMPLE_TEXT, provider, model)

    # 2. Texto real ambíguo do corpus.
    articles = load_corpus()
    target = None
    for a in articles:
        if "father of the internet" in a.title.lower():
            target = a
            break
    if target is None:
        target = articles[0]

    log.info("Artigo escolhido: uid=%s", target.uid)
    log.info("Titulo: %s", target.title)
    log.info("Tamanho full_text: %d chars\n", len(target.full_text))

    _classify_and_show_distribution(
        f"Texto real: {target.title[:60]}", target.full_text, provider, model
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
