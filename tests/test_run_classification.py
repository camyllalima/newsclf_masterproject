"""Tests for run_classification.py — pure functions only, no API calls."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from newsclf.config import LABELS_CNN18
from newsclf.corpus_loader import Article

# Make scripts/ importable.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

run_mod = importlib.import_module("run_classification")  # noqa: E402
_get_inputs = run_mod._get_inputs
_append_csv = run_mod._append_csv
_load_done_keys = run_mod._load_done_keys
_remove_invalid_single = run_mod._remove_invalid_single
_remove_invalid_percent = run_mod._remove_invalid_percent

# Default args namespace used by integration tests (no --provider).
_DEFAULT_ARGS = argparse.Namespace(
    provider=None, method="both", inputs=None, dry_run=False, workers=1,
)
# Single test provider so integration tests don't depend on real config.MODELS.
_TEST_MODELS = {"gemini": "test-model"}


_PROVIDER = "gemini"
_MODEL = "test-model"


def _single_path(tmp: Path):
    return lambda p: tmp / f"results_single_{p}.csv"


def _percent_path(tmp: Path):
    return lambda p: tmp / f"results_percent_{p}.csv"


def _single_row(uid: str, input_type: str, label: str | None) -> dict:
    """Build a single-method row with all expected columns."""
    return {
        "uid": uid,
        "provider": _PROVIDER,
        "model": _MODEL,
        "input_type": input_type,
        "predicted_label": label,
    }


def _percent_row(
    uid: str, input_type: str, *, valid: bool = True, top_label: str = "Business"
) -> dict:
    """Build a percent-method row with all expected columns."""
    row: dict = {
        "uid": uid,
        "provider": _PROVIDER,
        "model": _MODEL,
        "input_type": input_type,
    }
    if valid:
        row.update({lb: 0 for lb in LABELS_CNN18})
        row[top_label] = 100
        row["raw_response"] = None
    else:
        row.update({lb: None for lb in LABELS_CNN18})
        row["raw_response"] = "bad response"
    return row


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def full_article() -> Article:
    """Article with all four inputs populated."""
    return Article(
        uid="test-article-1",
        title="Test Article",
        full_text="This is the full text of the article.",
        highlights="These are the article highlights.",
        gold_standard="This is the gold standard summary.",
        chatgpt_summary="This is the ChatGPT summary.",
    )


@pytest.fixture()
def article_missing_gold() -> Article:
    """Article with empty gold_standard."""
    return Article(
        uid="test-article-2",
        title="Missing Gold",
        full_text="Some text.",
        highlights="Some highlights.",
        gold_standard="",
    )


@pytest.fixture()
def article_missing_highlights() -> Article:
    """Article with empty highlights."""
    return Article(
        uid="test-article-3",
        title="Missing Highlights",
        full_text="Some text.",
        highlights="",
        gold_standard="Gold text.",
    )


# ---------------------------------------------------------------------------
# _get_inputs
# ---------------------------------------------------------------------------


def test_get_inputs_all_present(full_article: Article) -> None:
    inputs = _get_inputs(full_article)
    assert set(inputs.keys()) == {
        "full_text", "extractive_summary", "abstractive_summary", "summary_chatgpt",
    }
    assert inputs["full_text"] == full_article.full_text
    assert inputs["extractive_summary"] == full_article.gold_standard
    assert inputs["abstractive_summary"] == full_article.highlights
    assert inputs["summary_chatgpt"] == full_article.chatgpt_summary


def test_get_inputs_skips_empty_gold(article_missing_gold: Article) -> None:
    inputs = _get_inputs(article_missing_gold)
    assert "extractive_summary" not in inputs
    assert "full_text" in inputs
    assert "abstractive_summary" in inputs


def test_get_inputs_skips_empty_highlights(article_missing_highlights: Article) -> None:
    inputs = _get_inputs(article_missing_highlights)
    assert "abstractive_summary" not in inputs
    assert "full_text" in inputs
    assert "extractive_summary" in inputs


def test_get_inputs_empty_full_text() -> None:
    article = Article(
        uid="empty-ft",
        title="Empty",
        full_text="",
        highlights="Some highlights.",
        gold_standard="Some gold.",
    )
    inputs = _get_inputs(article)
    assert "full_text" not in inputs
    assert "extractive_summary" in inputs
    assert "abstractive_summary" in inputs


# ---------------------------------------------------------------------------
# _load_done_keys
# ---------------------------------------------------------------------------


def test_load_done_keys_missing_file(tmp_path: Path) -> None:
    assert _load_done_keys(tmp_path / "nope.csv") == set()


def test_load_done_keys_reads_existing(tmp_path: Path) -> None:
    csv = tmp_path / "results.csv"
    csv.write_text("uid,input_type,predicted_label\nabc,full_text,X\nabc,extractive_summary,Y\n")
    assert _load_done_keys(csv) == {("abc", "full_text"), ("abc", "extractive_summary")}


def test_load_done_keys_ignores_extra_columns(tmp_path: Path) -> None:
    csv = tmp_path / "results.csv"
    csv.write_text("uid,provider,input_type,predicted_label\nabc,gemini,full_text,X\n")
    assert _load_done_keys(csv) == {("abc", "full_text")}


# ---------------------------------------------------------------------------
# _remove_invalid_single
# ---------------------------------------------------------------------------


def test_remove_invalid_single_removes_nan_rows(tmp_path: Path) -> None:
    csv = tmp_path / "single.csv"
    csv.write_text(
        "uid,input_type,predicted_label\n"
        "a,full_text,Business\n"
        "a,extractive_summary,\n"
        "b,full_text,Arts\n"
    )
    removed = _remove_invalid_single(csv)
    assert removed == 1
    df = pd.read_csv(csv)
    assert len(df) == 2
    assert set(df["uid"]) == {"a", "b"}


def test_remove_invalid_single_no_file(tmp_path: Path) -> None:
    assert _remove_invalid_single(tmp_path / "nope.csv") == 0


# ---------------------------------------------------------------------------
# _remove_invalid_percent
# ---------------------------------------------------------------------------


def test_remove_invalid_percent_removes_all_nan_labels(tmp_path: Path) -> None:
    csv = tmp_path / "percent.csv"
    labels = ["Arts", "Sports"]
    df = pd.DataFrame(
        [
            {"uid": "a", "input_type": "full_text", "Arts": 50, "Sports": 50},
            {"uid": "a", "input_type": "extractive_summary", "Arts": None, "Sports": None},
        ]
    )
    df.to_csv(csv, index=False)
    removed = _remove_invalid_percent(csv, labels)
    assert removed == 1
    result = pd.read_csv(csv)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# _append_csv
# ---------------------------------------------------------------------------


def test_append_csv_creates_with_header(tmp_path: Path) -> None:
    csv = tmp_path / "out.csv"
    df = pd.DataFrame({"a": [1], "b": [2]})
    _append_csv(df, csv)
    result = pd.read_csv(csv)
    assert list(result.columns) == ["a", "b"]
    assert len(result) == 1


def test_append_csv_appends_without_duplicate_header(tmp_path: Path) -> None:
    csv = tmp_path / "out.csv"
    df1 = pd.DataFrame({"a": [1], "b": [2]})
    df2 = pd.DataFrame({"a": [3], "b": [4]})
    _append_csv(df1, csv)
    _append_csv(df2, csv)
    result = pd.read_csv(csv)
    assert len(result) == 2
    assert list(result["a"]) == [1, 3]


# ---------------------------------------------------------------------------
# Integration: main() with mocked classify
# ---------------------------------------------------------------------------

_SINGLE_RESPONSE = json.dumps({"label": "Business"})
_PCT_VALUES = {lb: 0 for lb in LABELS_CNN18}
_PCT_VALUES["Business"] = 100
_PERCENT_RESPONSE = json.dumps(_PCT_VALUES)


def _fake_classify(_provider: str, _model: str, prompt: str) -> str:
    if '"label"' in prompt:
        return _SINGLE_RESPONSE
    return _PERCENT_RESPONSE


def _run_main_with_mocks(
    tmp_path: Path,
    articles: list[Article],
    fake_classify=_fake_classify,
) -> int:
    """Run main() with standard mocks. Return total classify() calls."""
    call_count = 0

    def counting_classify(p: str, m: str, prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        return fake_classify(p, m, prompt)

    with (
        patch.object(run_mod, "_parse_args", return_value=_DEFAULT_ARGS),
        patch.object(run_mod, "MODELS", _TEST_MODELS),
        patch.object(run_mod, "load_corpus", return_value=articles),
        patch.object(run_mod, "validate_corpus"),
        patch.object(run_mod, "classify", side_effect=counting_classify),
        patch.object(run_mod, "results_single_path", _single_path(tmp_path)),
        patch.object(run_mod, "results_percent_path", _percent_path(tmp_path)),
    ):
        run_mod.main()

    return call_count


def test_main_builds_correct_records(full_article: Article, tmp_path: Path) -> None:
    """Mock classify and verify that main() produces the expected records."""
    call_count = _run_main_with_mocks(tmp_path, [full_article])

    # 1 article × 4 inputs × 2 methods = 8 classify calls.
    assert call_count == 8

    df_s = pd.read_csv(tmp_path / "results_single_gemini.csv")
    assert len(df_s) == 4  # 4 inputs
    assert set(df_s["input_type"]) == {
        "full_text",
        "extractive_summary",
        "abstractive_summary",
        "summary_chatgpt",
    }
    assert (df_s["predicted_label"] == "Business").all()

    df_p = pd.read_csv(tmp_path / "results_percent_gemini.csv")
    assert len(df_p) == 4
    assert (df_p["Business"] == 100.0).all()
    assert df_p["raw_response"].isna().all()


# ---------------------------------------------------------------------------
# Granular done-detection tests
# ---------------------------------------------------------------------------


def test_valid_combo_is_skipped(full_article: Article, tmp_path: Path) -> None:
    """(a) All 4 input_types already valid in both CSVs → 0 API calls."""
    single_csv = tmp_path / "results_single_gemini.csv"
    percent_csv = tmp_path / "results_percent_gemini.csv"
    uid = full_article.uid

    itypes = ["full_text", "extractive_summary", "abstractive_summary", "summary_chatgpt"]
    pd.DataFrame([_single_row(uid, it, "Business") for it in itypes]).to_csv(
        single_csv, index=False
    )
    pd.DataFrame([_percent_row(uid, it) for it in itypes]).to_csv(percent_csv, index=False)

    call_count = _run_main_with_mocks(tmp_path, [full_article])
    assert call_count == 0

    # No extra rows appended.
    assert len(pd.read_csv(single_csv)) == 4
    assert len(pd.read_csv(percent_csv)) == 4


def test_invalid_combo_is_reprocessed(full_article: Article, tmp_path: Path) -> None:
    """(b) Row with NaN predicted_label is removed and reprocessed."""
    single_csv = tmp_path / "results_single_gemini.csv"
    percent_csv = tmp_path / "results_percent_gemini.csv"
    uid = full_article.uid
    itypes = ["full_text", "extractive_summary", "abstractive_summary", "summary_chatgpt"]

    # Single: full_text is invalid (None), others valid.
    rows_s = [
        _single_row(uid, "full_text", None),
        _single_row(uid, "extractive_summary", "Arts"),
        _single_row(uid, "abstractive_summary", "Arts"),
        _single_row(uid, "summary_chatgpt", "Arts"),
    ]
    pd.DataFrame(rows_s).to_csv(single_csv, index=False)

    # Percent: all valid.
    pd.DataFrame([_percent_row(uid, it) for it in itypes]).to_csv(percent_csv, index=False)

    call_count = _run_main_with_mocks(tmp_path, [full_article])

    # Only 1 single call for the invalid full_text; percent is done.
    assert call_count == 1

    df_s = pd.read_csv(single_csv)
    assert len(df_s) == 4  # 3 kept + 1 new
    tc_row = df_s[df_s["input_type"] == "full_text"]
    assert len(tc_row) == 1
    assert tc_row.iloc[0]["predicted_label"] == "Business"


def test_absent_combo_is_processed(full_article: Article, tmp_path: Path) -> None:
    """(c) Missing (uid, input_type) combo is processed."""
    single_csv = tmp_path / "results_single_gemini.csv"
    percent_csv = tmp_path / "results_percent_gemini.csv"
    uid = full_article.uid

    # Only full_text exists in both CSVs.
    pd.DataFrame([_single_row(uid, "full_text", "Arts")]).to_csv(single_csv, index=False)
    pd.DataFrame([_percent_row(uid, "full_text")]).to_csv(percent_csv, index=False)

    call_count = _run_main_with_mocks(tmp_path, [full_article])

    # 3 missing input_types × 2 methods = 6 calls.
    assert call_count == 6

    df_s = pd.read_csv(single_csv)
    assert len(df_s) == 4
    assert set(df_s["input_type"]) == {
        "full_text", "extractive_summary", "abstractive_summary", "summary_chatgpt",
    }


def test_partial_invalid_reprocesses_only_invalid(full_article: Article, tmp_path: Path) -> None:
    """(d) UID with 2 valid + 1 invalid: only the invalid one is reprocessed."""
    single_csv = tmp_path / "results_single_gemini.csv"
    percent_csv = tmp_path / "results_percent_gemini.csv"
    uid = full_article.uid

    # Single: extractive_summary has empty label.
    rows_s = [
        _single_row(uid, "full_text", "Arts"),
        _single_row(uid, "extractive_summary", ""),
        _single_row(uid, "abstractive_summary", "Sports"),
        _single_row(uid, "summary_chatgpt", "Health"),
    ]
    pd.DataFrame(rows_s).to_csv(single_csv, index=False)

    # Percent: abstractive_summary has all-NaN labels (invalid).
    rows_p = [
        _percent_row(uid, "full_text"),
        _percent_row(uid, "extractive_summary"),
        _percent_row(uid, "abstractive_summary", valid=False),
        _percent_row(uid, "summary_chatgpt"),
    ]
    pd.DataFrame(rows_p).to_csv(percent_csv, index=False)

    call_count = _run_main_with_mocks(tmp_path, [full_article])

    # single: 1 call (extractive_summary). percent: 1 call (abstractive_summary).
    assert call_count == 2

    df_s = pd.read_csv(single_csv)
    assert len(df_s) == 4
    # The reprocessed row should now have "Business".
    re_row = df_s[df_s["input_type"] == "extractive_summary"]
    assert re_row.iloc[0]["predicted_label"] == "Business"
    # Valid rows unchanged.
    assert df_s[df_s["input_type"] == "full_text"].iloc[0]["predicted_label"] == "Arts"
    assert df_s[df_s["input_type"] == "abstractive_summary"].iloc[0]["predicted_label"] == "Sports"

    df_p = pd.read_csv(percent_csv)
    assert len(df_p) == 4
    re_pct = df_p[df_p["input_type"] == "abstractive_summary"]
    assert re_pct.iloc[0]["Business"] == 100.0


def test_no_duplicates_after_reprocess(full_article: Article, tmp_path: Path) -> None:
    """(e) Reprocessing invalid rows does not create duplicates."""
    single_csv = tmp_path / "results_single_gemini.csv"
    percent_csv = tmp_path / "results_percent_gemini.csv"
    uid = full_article.uid
    itypes = ["full_text", "extractive_summary", "abstractive_summary", "summary_chatgpt"]

    # All 3 input_types invalid in single.
    pd.DataFrame([_single_row(uid, it, None) for it in itypes]).to_csv(single_csv, index=False)

    # All 3 valid in percent.
    pd.DataFrame([_percent_row(uid, it) for it in itypes]).to_csv(percent_csv, index=False)

    _run_main_with_mocks(tmp_path, [full_article])

    df_s = pd.read_csv(single_csv)
    df_p = pd.read_csv(percent_csv)

    # No duplicates: exactly 4 rows each (1 per input_type).
    assert len(df_s) == 4
    assert len(df_p) == 4
    assert df_s.duplicated(subset=["uid", "input_type"]).sum() == 0
    assert df_p.duplicated(subset=["uid", "input_type"]).sum() == 0
