"""Classify corpus articles with each LLM using two methods (single + percent).

Tracks progress granularly by (uid, input_type, method). Only combinations
that are missing or had invalid results are (re)processed — valid rows are
never repeated, even after partial failures.

Four inputs per article:
  - full_text:             article.full_text
  - extractive_summary:    article.gold_standard
  - abstractive_summary:   article.highlights
  - summary_chatgpt:       article.chatgpt_summary

Two classification methods per input:
  - single:  pick exactly one label  → results/results_single_<provider>.csv
  - percent: distribute 100% across all labels → results/results_percent_<provider>.csv

Usage:
    python scripts/run_classification.py                          # all providers
    python scripts/run_classification.py --provider chatgpt       # ChatGPT only
    python scripts/run_classification.py --provider gemini        # Gemini only
    python scripts/run_classification.py --method single          # single only
"""

from __future__ import annotations

import argparse
import logging
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from newsclf.config import (
    LABELS_CNN18,
    MODELS,
    results_percent_path,
    results_single_path,
)
from newsclf.corpus_loader import Article, load_corpus
from newsclf.data_validation import validate_corpus
from newsclf.llm_clients import classify
from newsclf.prompt import (
    build_prompt_percent,
    build_prompt_single,
    extract_label_single,
    extract_percent,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Control constants
# ---------------------------------------------------------------------------

LABELS = LABELS_CNN18

# Stable column order for the percent CSV.
PCT_COLS: list[str] = [
    "uid",
    "provider",
    "model",
    "input_type",
    *LABELS,
    "raw_response",
]

_INPUT_FIELDS: list[tuple[str, str]] = [
    ("full_text", "full_text"),
    ("extractive_summary", "gold_standard"),
    ("abstractive_summary", "highlights"),
    ("summary_chatgpt", "chatgpt_summary"),
    ("summary_deepseek", "deepseek_summary"),
    ("summary_gemini", "gemini_summary"),
]


# ---------------------------------------------------------------------------
# Granular done-detection helpers
# ---------------------------------------------------------------------------


def _remove_invalid_single(path: Path) -> int:
    """Remove rows with NaN/empty predicted_label. Return count removed."""
    if not path.exists():
        return 0
    df = pd.read_csv(path)
    valid = df["predicted_label"].notna() & (df["predicted_label"].astype(str).str.strip() != "")
    removed = int((~valid).sum())
    if removed:
        df[valid].to_csv(path, index=False)
        log.info("  Removed %d invalid single records.", removed)
    return removed


def _remove_invalid_percent(path: Path, labels: list[str]) -> int:
    """Remove rows where all label columns are NaN. Return count removed."""
    if not path.exists():
        return 0
    df = pd.read_csv(path)
    label_cols = [c for c in labels if c in df.columns]
    if not label_cols:
        return 0
    valid = df[label_cols].notna().any(axis=1)
    removed = int((~valid).sum())
    if removed:
        df[valid].to_csv(path, index=False)
        log.info("  Removed %d invalid percent records.", removed)
    return removed


def _load_done_keys(path: Path) -> set[tuple[str, str]]:
    """Return (uid, input_type) pairs present in a results CSV."""
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=["uid", "input_type"])
    return set(zip(df["uid"], df["input_type"]))


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _append_csv(df: pd.DataFrame, path: Path) -> None:
    """Append rows to a CSV, writing the header only if the file is new."""
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False)


def _get_inputs(
    article: Article,
    allowed: set[str] | None = None,
) -> dict[str, str]:
    """Return {input_type: text} for non-empty inputs of an article.

    If *allowed* is given, only input_types in that set are considered.
    """
    inputs: dict[str, str] = {}
    for input_type, attr in _INPUT_FIELDS:
        if allowed is not None and input_type not in allowed:
            continue
        value = getattr(article, attr)
        if value:
            inputs[input_type] = value
        else:
            log.warning("uid=%s: %s empty — skipping input.", article.uid, attr)
    return inputs


@dataclass
class _ArticleResult:
    """Thread-safe container for one article's classification output."""

    records_single: list[dict] = field(default_factory=list)
    records_percent: list[dict] = field(default_factory=list)
    inv_single: Counter = field(default_factory=Counter)
    inv_percent: Counter = field(default_factory=Counter)
    call_count: int = 0


def _classify_article(
    article: Article,
    provider: str,
    model: str,
    labels: list[str],
    done_single: set[tuple[str, str]],
    done_percent: set[tuple[str, str]],
    *,
    run_single: bool = True,
    run_percent: bool = True,
    allowed_inputs: set[str] | None = None,
) -> _ArticleResult:
    """Classify pending (uid, input_type) combos and return results."""
    inputs = _get_inputs(article, allowed=allowed_inputs)
    result = _ArticleResult()

    for input_type, text in inputs.items():
        key = (article.uid, input_type)
        need_single = run_single and key not in done_single
        need_percent = run_percent and key not in done_percent

        if not need_single and not need_percent:
            continue

        base = {
            "uid": article.uid,
            "provider": provider,
            "model": model,
            "input_type": input_type,
        }

        if need_single:
            prompt_s = build_prompt_single(text, labels)
            raw_s = classify(provider, model, prompt_s)
            predicted = extract_label_single(raw_s, labels)
            if predicted is None:
                result.inv_single[input_type] += 1
            result.records_single.append({**base, "predicted_label": predicted})
            result.call_count += 1

        if need_percent:
            prompt_p = build_prompt_percent(text, labels)
            raw_p = classify(provider, model, prompt_p)
            pct_dict = extract_percent(raw_p, labels)
            if pct_dict is None:
                result.inv_percent[input_type] += 1
                row_pct = {**base, **{lb: None for lb in labels}}
                row_pct["raw_response"] = raw_p
            else:
                row_pct = {**base, **pct_dict}
                row_pct["raw_response"] = None
            result.records_percent.append(row_pct)
            result.call_count += 1

    return result


def _flush_results(
    result: _ArticleResult,
    single_csv: Path,
    percent_csv: Path,
    csv_lock: threading.Lock,
) -> None:
    """Write article results to CSV files under a lock."""
    with csv_lock:
        if result.records_single:
            _append_csv(pd.DataFrame(result.records_single), single_csv)
        if result.records_percent:
            _append_csv(pd.DataFrame(result.records_percent)[PCT_COLS], percent_csv)


def _log_run_report(
    total_calls: int,
    invalid_single: Counter[str],
    invalid_percent: Counter[str],
) -> None:
    """Log a summary of the classification run."""
    log.info("")
    log.info("=== Run summary ===")
    log.info("API calls:          %d", total_calls)

    log.info("")
    log.info("=== Invalid (None) — single ===")
    for it in sorted(invalid_single):
        log.info("  %-20s %d", it, invalid_single[it])
    if not invalid_single:
        log.info("  none")

    log.info("=== Invalid (None) — percent ===")
    for it in sorted(invalid_percent):
        log.info("  %-20s %d", it, invalid_percent[it])
    if not invalid_percent:
        log.info("  none")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify corpus articles with LLMs.")
    parser.add_argument(
        "--provider",
        choices=list(MODELS.keys()),
        default=None,
        help="Run only this provider. Default: all providers in config.MODELS.",
    )
    parser.add_argument(
        "--method",
        choices=["single", "percent", "both"],
        default="both",
        help="Run only this method. Default: both.",
    )
    all_input_names = [name for name, _ in _INPUT_FIELDS]
    parser.add_argument(
        "--inputs",
        default=None,
        help=(
            "Comma-separated list of input_types to process. "
            f"Available: {','.join(all_input_names)}. Default: all."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count pending calls and exit without calling any API.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for API calls. Default: 1 (sequential).",
    )
    return parser.parse_args()


def main() -> None:
    """Load corpus, classify pending (uid, input_type) combos, and append."""
    args = _parse_args()

    articles = load_corpus()
    validate_corpus(articles)
    log.info("Corpus: %d articles.", len(articles))

    if args.provider:
        models_to_run = {args.provider: MODELS[args.provider]}
    else:
        models_to_run = MODELS

    # Resolve allowed input types.
    all_input_names = {name for name, _ in _INPUT_FIELDS}
    if args.inputs:
        allowed_inputs = set(args.inputs.split(","))
        unknown = allowed_inputs - all_input_names
        if unknown:
            raise SystemExit(f"Unknown input types: {unknown}. Available: {all_input_names}")
    else:
        allowed_inputs = None  # means "all"

    run_single = args.method in ("single", "both")
    run_percent = args.method in ("percent", "both")

    invalid_single: Counter[str] = Counter()
    invalid_percent: Counter[str] = Counter()
    total_calls = 0

    for provider, model in models_to_run.items():
        single_csv = results_single_path(provider)
        percent_csv = results_percent_path(provider)

        # 1. Remove invalid rows so they can be reprocessed.
        if run_single:
            _remove_invalid_single(single_csv)
        if run_percent:
            _remove_invalid_percent(percent_csv, LABELS)

        # 2. Load valid (uid, input_type) keys per method.
        done_single = _load_done_keys(single_csv) if run_single else set()
        done_percent = _load_done_keys(percent_csv) if run_percent else set()

        # 3. Count pending work and filter articles.
        pending_single = 0
        pending_percent = 0
        articles_with_work: list[Article] = []

        for article in articles:
            inputs = _get_inputs(article, allowed=allowed_inputs)
            has_work = False
            for input_type in inputs:
                key = (article.uid, input_type)
                if run_single and key not in done_single:
                    pending_single += 1
                    has_work = True
                if run_percent and key not in done_percent:
                    pending_percent += 1
                    has_work = True
            if has_work:
                articles_with_work.append(article)

        total_pending = pending_single + pending_percent
        log.info(
            "[%s] Pending: %d single + %d percent = %d calls (%d articles).",
            provider,
            pending_single,
            pending_percent,
            total_pending,
            len(articles_with_work),
        )

        if not total_pending:
            log.info("[%s] All complete!", provider)
            continue

        if args.dry_run:
            log.info("[%s] DRY RUN — would make %d API calls. Exiting.", provider, total_pending)
            total_calls += total_pending
            continue

        progress = tqdm(total=total_pending, desc=f"classifying [{provider}]")
        csv_lock = threading.Lock()

        classify_kwargs = dict(
            provider=provider,
            model=model,
            labels=LABELS,
            done_single=done_single,
            done_percent=done_percent,
            run_single=run_single,
            run_percent=run_percent,
            allowed_inputs=allowed_inputs,
        )

        def _process_and_flush(article: Article) -> _ArticleResult:
            result = _classify_article(article, **classify_kwargs)
            _flush_results(result, single_csv, percent_csv, csv_lock)
            return result

        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(_process_and_flush, article): article
                    for article in articles_with_work
                }
                for future in as_completed(futures):
                    result = future.result()
                    invalid_single += result.inv_single
                    invalid_percent += result.inv_percent
                    progress.update(result.call_count)
        else:
            for article in articles_with_work:
                result = _process_and_flush(article)
                invalid_single += result.inv_single
                invalid_percent += result.inv_percent
                progress.update(result.call_count)

        progress.close()
        total_calls += total_pending

    _log_run_report(total_calls, invalid_single, invalid_percent)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
