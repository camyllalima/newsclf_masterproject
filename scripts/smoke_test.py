"""Smoke test: validate single-label API calls before running the full pipeline.

Sends a short sports text to each active provider and checks if the response
parses into a valid label.

Usage:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import logging

from newsclf.config import MODELS
from newsclf.llm_clients import classify
from newsclf.prompt import build_prompt_single, extract_label_single

log = logging.getLogger(__name__)

SAMPLE_TEXT = (
    "Manchester City defeated Arsenal 3-1 in a thrilling Premier League match. "
    "Haaland scored twice in the first half, sealing the victory for the home side. "
    "Arsenal's manager said the team needs to improve its defensive strategy."
)

SAMPLE_LABELS = ["Sports", "Government and Politics", "Business", "Health"]


def main() -> None:
    prompt = build_prompt_single(SAMPLE_TEXT, SAMPLE_LABELS)
    valid_count = 0

    for provider, model in MODELS.items():
        log.info("--- %s (%s) ---", provider, model)
        try:
            raw = classify(provider, model, prompt)
            label = extract_label_single(raw, SAMPLE_LABELS)
            log.info("Raw response:   %s", raw)
            if label is not None:
                log.info("Extracted label: %s", label)
                valid_count += 1
            else:
                log.warning("Invalid or unrecognised label in response.")
        except Exception as exc:
            log.error("Error calling %s: %s: %s", provider, type(exc).__name__, exc)

    log.info("\n=== Summary ===")
    log.info("%d/%d providers returned a valid label.", valid_count, len(MODELS))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
