"""Build classification prompts and parse model answers."""

from __future__ import annotations

import json
import re

from newsclf.config import LABEL_DESCRIPTIONS


def _strip_code_fences(text: str | None) -> str | None:
    """Remove markdown code fences (```json ... ``` or ``` ... ```) if present."""
    if text is None:
        return None
    return re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), flags=re.IGNORECASE).rstrip("`").strip()


def _build_label_index(labels: list[str]) -> dict[str, str]:
    """Build a lookup mapping lowercase canonical and abbreviated forms to canonical labels.

    Covers: exact match, lowercase, first word (e.g. "Employment" -> "Employment and Work").
    """
    index: dict[str, str] = {}
    for label in labels:
        index[label.lower()] = label
        first_word = label.split(" and ")[0].strip().lower()
        if first_word != label.lower():
            index.setdefault(first_word, label)
    return index


def _resolve_label(raw: str, labels: list[str], index: dict[str, str] | None = None) -> str | None:
    """Try to match *raw* to a canonical label (exact, case-insensitive, prefix)."""
    stripped = raw.strip()
    if stripped in labels:
        return stripped
    if index is None:
        index = _build_label_index(labels)
    return index.get(stripped.lower())


def label_block(labels: list[str]) -> str:
    """Format labels with their descriptions for inclusion in prompts."""
    lines: list[str] = []
    for label in labels:
        desc = LABEL_DESCRIPTIONS.get(label, "")
        if desc:
            lines.append(f"- {label}: {desc}")
        else:
            lines.append(f"- {label}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Single-label classification
# ---------------------------------------------------------------------------

SINGLE_TEMPLATE = """\
You are an expert news classifier. Your task is to read a news article and \
assign it to exactly ONE of the categories listed below.

## Categories

{label_block}

## Rules

1. Choose the SINGLE category that best describes the PRIMARY topic of the article.
2. If the article touches multiple topics, pick the MOST DOMINANT one.
3. Use ONLY a category from the list above — do NOT invent new categories.
4. Do NOT include any explanation, commentary, or extra text.
5. Respond with valid JSON and nothing else.

## Output format

Respond ONLY with a JSON object in this exact format:
{{"label": "<one of the categories above>"}}

## Article

\"\"\"{text}\"\"\""""


def build_prompt_single(text: str, labels: list[str]) -> str:
    """Return a prompt that asks the model to pick exactly one label."""
    return SINGLE_TEMPLATE.format(
        label_block=label_block(labels),
        text=text,
    )


def extract_label_single(response_text: str, labels: list[str]) -> str | None:
    """Return the valid label from the model's JSON answer, or None if invalid."""
    cleaned = _strip_code_fences(response_text)
    if cleaned is None:
        return None
    index = _build_label_index(labels)

    # Try JSON first.
    try:
        parsed = json.loads(cleaned)
        raw_label = parsed.get("label") or parsed.get("category", "")
        resolved = _resolve_label(str(raw_label), labels, index)
        if resolved:
            return resolved
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    # Fallback: treat the whole response as a bare label string.
    return _resolve_label(cleaned, labels, index)


# ---------------------------------------------------------------------------
# Percentage-distribution classification
# ---------------------------------------------------------------------------

_PERCENT_TEMPLATE = """\
You are an expert news classifier. Your task is to read a news article and \
estimate how relevant each of the categories below is to the article, \
distributing exactly 100 percentage points across ALL categories.

## Categories

{label_block}

## Rules

1. Assign a percentage (integer from 0 to 100) to EVERY category.
2. The percentages MUST sum to exactly 100.
3. Include ALL categories in your response, even those with 0%.
4. Use ONLY the category names listed above — do NOT invent new ones.
5. Do NOT include any explanation, commentary, or extra text.
6. Respond with valid JSON and nothing else.

## Output format

Respond ONLY with a JSON object mapping every category to its percentage:
{{"Arts": 0, "Business": 85, "Health": 15, ...}}

## Article

\"\"\"{text}\"\"\""""


def build_prompt_percent(text: str, labels: list[str]) -> str:
    """Return a prompt that asks the model to distribute 100% across all labels."""
    return _PERCENT_TEMPLATE.format(
        label_block=label_block(labels),
        text=text,
    )


def _parse_text_list(text: str) -> dict[str, str]:
    """Parse a text-list response like '- Arts: 0\\n- Business: 40' into a raw dict."""
    pairs: dict[str, str] = {}
    for m in re.finditer(r"[•\-*]?\s*([A-Za-z][\w\s&]+?):\s*(\d+)", text):
        pairs[m.group(1).strip()] = m.group(2)
    return pairs


def _normalize_parsed_dict(
    parsed: dict,
    labels: list[str],
    index: dict[str, str],
) -> dict[str, float] | None:
    """Normalize a parsed dict: resolve label names, fill missing with 0."""
    result: dict[str, float] = {}
    # Map parsed keys to canonical labels.
    for key, val in parsed.items():
        resolved = _resolve_label(str(key), labels, index)
        if resolved is None:
            continue
        try:
            result[resolved] = float(val)
        except (ValueError, TypeError):
            return None
    # Fill missing labels with 0.
    for label in labels:
        result.setdefault(label, 0.0)
    return result


def extract_percent(
    response_text: str,
    labels: list[str],
) -> dict[str, float] | None:
    """Parse the model's JSON answer into {label: percentage}, or None if invalid."""
    cleaned = _strip_code_fences(response_text)
    if cleaned is None:
        return None
    index = _build_label_index(labels)

    # Try JSON first.
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            result = _normalize_parsed_dict(parsed, labels, index)
            if result is not None:
                return result
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: parse text-list format ("- Arts: 0\n- Business: 40\n...").
    pairs = _parse_text_list(cleaned)
    if pairs:
        return _normalize_parsed_dict(pairs, labels, index)

    return None
