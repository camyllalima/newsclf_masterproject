"""Unit tests for the pure functions in prompt.py."""

import json

from newsclf.prompt import (
    build_prompt_percent,
    build_prompt_single,
    extract_label_single,
    extract_percent,
)

LABELS = ["Sports", "Business", "Health"]


# ---------------------------------------------------------------------------
# Single-label
# ---------------------------------------------------------------------------


def test_build_prompt_single_lists_all_labels() -> None:
    prompt = build_prompt_single("some news text", LABELS)
    for label in LABELS:
        assert label in prompt


def test_extract_label_single_accepts_valid_json() -> None:
    assert extract_label_single('{"label": "Sports"}', LABELS) == "Sports"


def test_extract_label_single_rejects_unknown_label() -> None:
    assert extract_label_single('{"label": "Politics"}', LABELS) is None


def test_extract_label_single_handles_invalid_json() -> None:
    assert extract_label_single("not valid json", LABELS) is None


# ---------------------------------------------------------------------------
# Percentage-distribution
# ---------------------------------------------------------------------------


def test_build_prompt_percent_lists_all_labels() -> None:
    prompt = build_prompt_percent("some news text", LABELS)
    for label in LABELS:
        assert label in prompt
    assert "100" in prompt


def test_extract_percent_valid() -> None:
    response = json.dumps({"Sports": 70, "Business": 20, "Health": 10})
    result = extract_percent(response, LABELS)
    assert result is not None
    assert result["Sports"] == 70.0
    assert result["Business"] == 20.0
    assert result["Health"] == 10.0


def test_extract_percent_missing_label_filled_with_zero() -> None:
    response = json.dumps({"Sports": 70, "Business": 30})
    result = extract_percent(response, LABELS)
    assert result is not None
    assert result["Health"] == 0.0
    assert result["Sports"] == 70.0


def test_extract_percent_invalid_json() -> None:
    assert extract_percent("not json", LABELS) is None


def test_extract_percent_non_numeric_value() -> None:
    response = json.dumps({"Sports": "a lot", "Business": 20, "Health": 10})
    assert extract_percent(response, LABELS) is None


# ---------------------------------------------------------------------------
# Code-fence stripping
# ---------------------------------------------------------------------------


def test_extract_label_single_with_json_fence() -> None:
    response = '```json\n{"label": "Sports"}\n```'
    assert extract_label_single(response, LABELS) == "Sports"


def test_extract_label_single_with_plain_fence() -> None:
    response = '```\n{"label": "Business"}\n```'
    assert extract_label_single(response, LABELS) == "Business"


def test_extract_label_single_without_fence() -> None:
    assert extract_label_single('{"label": "Health"}', LABELS) == "Health"


def test_extract_percent_with_json_fence() -> None:
    body = json.dumps({"Sports": 50, "Business": 30, "Health": 20})
    response = f"```json\n{body}\n```"
    result = extract_percent(response, LABELS)
    assert result is not None
    assert result["Sports"] == 50.0


def test_extract_percent_with_plain_fence() -> None:
    body = json.dumps({"Sports": 50, "Business": 30, "Health": 20})
    response = f"```\n{body}\n```"
    result = extract_percent(response, LABELS)
    assert result is not None
    assert result["Business"] == 30.0


def test_extract_percent_without_fence() -> None:
    body = json.dumps({"Sports": 50, "Business": 30, "Health": 20})
    result = extract_percent(body, LABELS)
    assert result is not None
    assert result["Health"] == 20.0


# ---------------------------------------------------------------------------
# Fuzzy / recovery parsing
# ---------------------------------------------------------------------------

FULL_LABELS = [
    "Arts",
    "Business",
    "Economy and Finance",
    "Employment and Work",
    "Government and Politics",
    "Health",
    "Society and Culture",
    "Sports",
]


def test_extract_label_single_bare_text() -> None:
    assert extract_label_single("Economy and Finance", FULL_LABELS) == "Economy and Finance"


def test_extract_label_single_bare_text_with_whitespace() -> None:
    assert extract_label_single("  Sports  ", FULL_LABELS) == "Sports"


def test_extract_label_single_abbreviated() -> None:
    assert extract_label_single('{"label": "Economy"}', FULL_LABELS) == "Economy and Finance"


def test_extract_label_single_abbreviated_bare() -> None:
    assert extract_label_single("Society", FULL_LABELS) == "Society and Culture"


def test_extract_label_single_category_key() -> None:
    assert (
        extract_label_single('{"category": "Employment and Work"}', FULL_LABELS)
        == "Employment and Work"
    )


def test_extract_label_single_none_response() -> None:
    assert extract_label_single(None, FULL_LABELS) is None


def test_extract_percent_abbreviated_labels() -> None:
    response = json.dumps({"Sports": 50, "Economy": 30, "Health": 20})
    result = extract_percent(response, FULL_LABELS)
    assert result is not None
    assert result["Economy and Finance"] == 30.0
    assert result["Arts"] == 0.0


def test_extract_percent_text_list_format() -> None:
    response = "- Sports: 50\n- Business: 30\n- Health: 20"
    result = extract_percent(response, FULL_LABELS)
    assert result is not None
    assert result["Sports"] == 50.0
    assert result["Business"] == 30.0
    assert result["Arts"] == 0.0


def test_extract_percent_none_response() -> None:
    assert extract_percent(None, FULL_LABELS) is None
