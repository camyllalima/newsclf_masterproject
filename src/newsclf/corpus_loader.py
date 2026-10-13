"""Load the CNN corpus from data/db3/ (plain-text format)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from newsclf.config import DB3_DIR, SUMMARIES_DEEPSEEK_DIR, SUMMARIES_DIR, SUMMARIES_GEMINI_DIR

log = logging.getLogger(__name__)


@dataclass
class Article:
    """A single CNN news article with its text and summaries."""

    uid: str  # filename normalised (lowercase alphanumerics only)
    title: str  # original filename without .txt
    full_text: str
    highlights: str
    gold_standard: str
    chatgpt_summary: str = ""
    deepseek_summary: str = ""
    gemini_summary: str = ""


def clean_text(text: str) -> str:
    """Replace residual XML entities and normalise whitespace."""
    text = text.replace("&apost;", "'").replace("&quot;", '"')
    return re.sub(r"\s+", " ", text).strip()


def _filename_to_uid(name: str) -> str:
    """Reduce a filename to lowercase alphanumerics only."""
    name = name.lower().replace(".txt", "")
    return re.sub(r"[^a-z0-9]", "", name)


def _read_txt(path: Path) -> str:
    """Read a .txt file, join lines with space, and clean."""
    with open(path, encoding="utf-8", errors="replace") as f:
        return clean_text(" ".join(line.strip() for line in f if line.strip()))


def load_corpus(data_dir: str | Path = DB3_DIR) -> list[Article]:
    """Load all articles from db3 plain-text directories."""
    base = Path(data_dir)
    texts_dir = base / "texts"
    gs_dir = base / "goldstandards"
    hl_dir = base / "highlights"

    if not texts_dir.is_dir():
        raise FileNotFoundError(f"Texts directory not found: {texts_dir}")

    txt_files = sorted(f for f in texts_dir.iterdir() if f.suffix == ".txt")
    articles: list[Article] = []
    missing_gs: list[str] = []
    missing_hl: list[str] = []

    for txt_path in txt_files:
        fname = txt_path.name
        title = txt_path.stem
        uid = _filename_to_uid(fname)

        full_text = _read_txt(txt_path)

        gs_path = gs_dir / fname
        if gs_path.exists():
            gold_standard = _read_txt(gs_path)
        else:
            gold_standard = ""
            missing_gs.append(fname)

        hl_path = hl_dir / fname
        if hl_path.exists():
            highlights = _read_txt(hl_path)
        else:
            highlights = ""
            missing_hl.append(fname)

        articles.append(
            Article(
                uid=uid,
                title=title,
                full_text=full_text,
                highlights=highlights,
                gold_standard=gold_standard,
            )
        )

    # --- Load external summaries (optional) ---
    uid_to_article = {a.uid: a for a in articles}

    n_summaries = 0
    if SUMMARIES_DIR.is_dir():
        for summary_path in SUMMARIES_DIR.glob("*.txt"):
            uid = _filename_to_uid(summary_path.name)
            art = uid_to_article.get(uid)
            if art is not None:
                art.chatgpt_summary = summary_path.read_text(encoding="utf-8").strip()
                n_summaries += 1

    n_deepseek_summaries = 0
    if SUMMARIES_DEEPSEEK_DIR.is_dir():
        for summary_path in SUMMARIES_DEEPSEEK_DIR.glob("*.txt"):
            uid = _filename_to_uid(summary_path.name)
            art = uid_to_article.get(uid)
            if art is not None:
                art.deepseek_summary = summary_path.read_text(encoding="utf-8").strip()
                n_deepseek_summaries += 1

    n_gemini_summaries = 0
    if SUMMARIES_GEMINI_DIR.is_dir():
        for summary_path in SUMMARIES_GEMINI_DIR.glob("*.txt"):
            uid = _filename_to_uid(summary_path.name)
            art = uid_to_article.get(uid)
            if art is not None:
                art.gemini_summary = summary_path.read_text(encoding="utf-8").strip()
                n_gemini_summaries += 1

    # --- Report ---
    log.info("=== Loading report ===")
    log.info("Articles loaded:               %d", len(articles))
    log.info("Missing gold_standard:         %d", len(missing_gs))
    log.info("Missing highlights:            %d", len(missing_hl))
    log.info("ChatGPT summaries loaded:      %d", n_summaries)
    log.info("DeepSeek summaries loaded:     %d", n_deepseek_summaries)
    log.info("Gemini summaries loaded:       %d", n_gemini_summaries)

    # Verify uid uniqueness.
    uids = [a.uid for a in articles]
    if len(uids) != len(set(uids)):
        log.warning("WARNING: UIDs are not unique! Check file names.")
    else:
        log.info("UIDs únicos confirmados:        %d", len(uids))

    return articles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    corpus = load_corpus()

    if corpus:
        a = corpus[0]
        log.info("\n=== Example (first article) ===")
        log.info("uid:            %s", a.uid)
        log.info("title:          %s", a.title)
        log.info("full_text size: %d chars", len(a.full_text))
        log.info("highlights size:%d chars", len(a.highlights))
        log.info("gold_standard:  %d chars", len(a.gold_standard))
        if a.gold_standard:
            log.info("  preview: %s...", a.gold_standard[:120])
