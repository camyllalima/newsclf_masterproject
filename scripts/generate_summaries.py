"""Generate abstractive summaries of corpus articles using any LLM provider.

For each article, sends the full_text with a fixed summarisation prompt
and saves the summary to data/summaries_<provider>/<uid>.txt.

Uses the existing cache/retry/delay infrastructure from llm_clients.

Usage:
    python scripts/generate_summaries.py --provider chatgpt
    python scripts/generate_summaries.py --provider deepseek --limit 10
    python scripts/generate_summaries.py --provider gemini --workers 4
    python scripts/generate_summaries.py --provider chatgpt --qc-only
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from newsclf.config import MODELS, SUMMARY_PROMPT_TEMPLATE, summaries_csv, summaries_dir
from newsclf.corpus_loader import load_corpus
from newsclf.llm_clients import classify
from newsclf.text_utils import count_sentences

log = logging.getLogger(__name__)

VALID_PROVIDERS = list(MODELS.keys())


def _build_csv(provider: str) -> None:
    """(Re)build the consolidated CSV from all .txt files."""
    out_dir = summaries_dir(provider)
    if not out_dir.exists():
        return
    files = sorted(out_dir.glob("*.txt"))
    if not files:
        return
    rows = [
        {"uid": f.stem, "summary": f.read_text(encoding="utf-8").strip()}
        for f in files
    ]
    csv_path = summaries_csv(provider)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    log.info("CSV rebuilt: %s (%d rows)", csv_path, len(rows))


def _run_qc(provider: str) -> None:
    """Count sentences of all generated summaries and report distribution."""
    out_dir = summaries_dir(provider)
    if not out_dir.exists():
        log.info("No summaries found in %s", out_dir)
        return

    files = sorted(out_dir.glob("*.txt"))
    log.info("Summaries generated: %d\n", len(files))

    counts: list[int] = []
    for f in files:
        text = f.read_text(encoding="utf-8").strip()
        counts.append(count_sentences(text))

    if not counts:
        return

    arr = np.array(counts)
    log.info("--- Sentence statistics ---")
    log.info("  Mean:    %.1f", arr.mean())
    log.info("  Median:  %.1f", np.median(arr))
    log.info("  Min:     %d", arr.min())
    log.info("  Max:     %d", arr.max())
    log.info("")

    in_target = int(((arr >= 3) & (arr <= 4)).sum())
    pct = 100 * in_target / len(arr)
    log.info("In target (3-4): %d/%d (%.1f%%)", in_target, len(arr), pct)
    log.info("")

    log.info("--- Distribution ---")
    for k in range(1, 7):
        n = int((arr == k).sum())
        log.info("  %d sentence(s): %4d (%.1f%%)", k, n, 100 * n / len(arr))
    n_7plus = int((arr >= 7).sum())
    if n_7plus:
        log.info("  7+:             %4d (%.1f%%)", n_7plus, 100 * n_7plus / len(arr))

    _build_csv(provider)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate abstractive summaries.")
    parser.add_argument(
        "--provider",
        required=True,
        choices=VALID_PROVIDERS,
        help="LLM provider to use.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only first N articles.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers. Default: 1.")
    parser.add_argument("--qc-only", action="store_true", help="Only run QC on existing summaries.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    provider: str = args.provider
    model = MODELS[provider]
    out_dir = summaries_dir(provider)

    if args.qc_only:
        _run_qc(provider)
        return

    articles = load_corpus()
    if args.limit:
        articles = articles[: args.limit]

    log.info("Provider: %s (%s)", provider, model)
    log.info("Articles to process: %d", len(articles))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find which are already done (incremental).
    already_done = {f.stem for f in out_dir.glob("*.txt")}
    pending = [a for a in articles if a.uid not in already_done]
    log.info("Already done: %d, pending: %d\n", len(articles) - len(pending), len(pending))

    if not pending:
        log.info("Nothing to do.")
        _run_qc(provider)
        return

    def _summarise_one(article):
        prompt = SUMMARY_PROMPT_TEMPLATE.format(text=article.full_text)
        summary = classify(provider, model, prompt)
        path = out_dir / f"{article.uid}.txt"
        path.write_text(summary.strip(), encoding="utf-8")

    progress = tqdm(total=len(pending), desc=f"summarising [{provider}]")

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_summarise_one, a): a for a in pending}
            for future in as_completed(futures):
                future.result()
                progress.update(1)
    else:
        for article in pending:
            _summarise_one(article)
            progress.update(1)

    progress.close()

    log.info("\nSummaries saved to %s\n", out_dir)
    _run_qc(provider)  # also rebuilds CSV


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
