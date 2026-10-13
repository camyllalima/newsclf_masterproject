"""Text utilities shared across scripts (sentence splitting, etc.)."""

from __future__ import annotations

import re

# Abbreviations whose period should NOT trigger a sentence boundary.
_ABBREV_RE = re.compile(
    r"\b("
    r"Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|approx|dept|est|vol"
    r"|Gen|Gov|Sgt|Cpl|Pvt|Lt|Col|Capt|Maj|Rev|Hon"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
    r"|St|Ave|Blvd|Inc|Corp|Ltd|Co"
    r")\.",
    re.IGNORECASE,
)

# Dotted abbreviations like U.S., U.N., U.K., E.U.
_DOTTED_ABBREV_RE = re.compile(r"\b([A-Z]\.){2,}")

# Sentence boundary: .!? (optionally followed by closing quote) then
# whitespace, then uppercase letter, opening quote, or CNN "NEW:" marker.
_SENT_SPLIT = re.compile(
    r'(?<=[.!?])'
    r'(?:["\u201d])?\s+'
    r'(?=[A-Z"\u201c\u2018(]|NEW:)'
)

_PH = "\x00"


def count_sentences(text: str) -> int:
    """Count sentences in a text using regex splitting."""
    if not text or not text.strip():
        return 0
    # Mask abbreviation periods so they don't trigger splits.
    masked = _DOTTED_ABBREV_RE.sub(lambda m: m.group().replace(".", _PH), text)
    masked = _ABBREV_RE.sub(lambda m: m.group().replace(".", _PH), masked)
    # Also mask decimal numbers (e.g. 3.5).
    masked = re.sub(r"(\d)\.(\d)", lambda m: f"{m.group(1)}{_PH}{m.group(2)}", masked)
    parts = _SENT_SPLIT.split(masked.strip())
    return len([p for p in parts if p.strip()])
