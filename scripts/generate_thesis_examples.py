"""Generate LaTeX snippets for thesis example sections from project data.

Run from the project root (where src/newsclf, data/ and results/ live):

    python scripts/generate_thesis_examples.py --output ../dissertacao_v2

Generated files (overwrite any provisional drafts):

    refs/ref_fulltext.tex
    refs/ref_goldstandard.tex
    refs/ref_highlights.tex
    refs/ref_llm_summaries.tex
    refs/ref_answer_key.tex
    appendix/examples_per_class.tex
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from newsclf.answer_key import (
    ANNOTATOR_COLS,
    ANNOTATOR_NAMES,
    XLSX_PATH,
    canonicalise_label,
    compute_ranking,
)
from newsclf.config import (
    LABELS_CNN18,
    results_percent_path,
    results_single_path,
    summaries_dir,
)
from newsclf.corpus_loader import Article, load_corpus

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROVIDERS = ["chatgpt", "deepseek", "gemini"]
PROVIDER_DISPLAY: dict[str, str] = {
    "chatgpt": "ChatGPT",
    "deepseek": "DeepSeek",
    "gemini": "Gemini",
}

# The four input types used per model in the thesis examples.
# "own_summary" maps to summary_<provider> at runtime.
INPUT_TYPES = ["full_text", "extractive_summary", "abstractive_summary", "own_summary"]
INPUT_DISPLAY: dict[str, str] = {
    "full_text": "Full text",
    "extractive_summary": "Extr. summary",
    "abstractive_summary": "Abstr. summary",
    "own_summary": "Own summary",
}

REFERENCE_UID = "10shockingmedicalmistakes"

# One example per category — same articles as Appendices A–R in Silva (2020).
EXAMPLES_BY_CATEGORY: dict[str, str] = {
    "Arts": "fromthevaultvintageworldsfairphotos",
    "Automotive": "ferraridesignercreatessuperluxurioustrain",
    "Business": "japansignsdealwithturkeytobuildnuclearplant",
    "Computer and Internet": "facebookswhatsappaddingvoicecalls",
    "Economy and Finance": "cypriotdisastrousdecisionhowtoturnadramaintoacrisis",
    "Education": "pakistaniteeninspiresotherstofightforeducation",
    "Employment and Work": "unemploymentineuropeisthereasolution", # do Ihago: "ukpolicearrest4onsuspicionofsyrialinkedterroroffenses"  (top-1 ponderado: Society and Culture)
    "Entertainment": "facebookaddsnewwaytotagupdatesaboutmusictv",
    "Food and Beverage": "poshnycrestaurantroastedbyhealthinspectors",
    "Games": "curitibastadiumretainsworldcupstatus",
    "Government and Politics": "usairstrikeaimsatisisnearbaghdad",
    "Health": "eatnutslivelonger",
    "Law": "prosecutorssayaaronhernandezdiscussedkillinginjailhousecalls",
    "Science": "spaceweatherfinewithachanceofsolarflares",
    "Society and Culture": "missingamericanjournalistsparentssendoursonhomefromsyriaforchristmas",
    "Sports": "barcelonaboostedbytitovilanovareturn",
    "Travel": "toplesstravelphotoslatesttravelrage",
    "Weather": "packing105mphwindshurricanecarlottabearsdownonmexicocoast", # do Ihago: "climatechangecouldcost100billionayear"  (top-1 ponderado: Government and Politics)
}


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------


# Abbreviations that should NOT trigger sentence breaks.
_ABBREVIATIONS: set[str] = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sen",
    "rep",
    "gov",
    "gen",
    "col",
    "lt",
    "sgt",
    "capt",
    "cmdr",
    "adm",
    "st",
    "jr",
    "sr",
    "rev",
    "fr",
    "no",
    "vs",
    "inc",
    "co",
    "corp",
    "ltd",
    "dept",
    "univ",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
    "u.s",
    "u.k",
    "u.n",
    "e.u",
    "d.c",
    "u.s.a",
    "a.m",
    "p.m",
    "e.g",
    "i.e",
    "etc",
    "approx",
    "est",
    "mt",
    "ft",
}

# Sentence boundary: .!? optionally followed by closing quotes/parens,
# whitespace, then an opening quote/paren or uppercase letter/digit.
_SENTENCE_END = re.compile(r"[.!?][\"'\u201d\u2019)\]]*\s+(?=[\"'\u201c\u2018(\[]?[A-Z0-9])")


def escape_tex(s: str) -> str:
    """Escape special LaTeX characters and convert quotes by context."""
    s = (
        s.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2014", "---")
        .replace("\u2013", "--")
    )
    if "\ufffd" in s:
        log.warning("Corrupted character (U+FFFD) in corpus text: %.80s", s)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\u20ac": r"\texteuro{}",
        "\u00a5": r"\textyen{}",
        "\u00a3": r"\pounds{}",
    }
    s = "".join(replacements.get(c, c) for c in s)
    # Decide quote direction by preceding character (not alternation).
    out: list[str] = []
    for i, c in enumerate(s):
        if c != '"':
            out.append(c)
            continue
        prev = s[i - 1] if i > 0 else " "
        out.append("``" if prev.isspace() or prev in "([{-" else "''")
    return "".join(out)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, handling abbreviations and quoted endings."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences: list[str] = []
    start = 0
    for m in _SENTENCE_END.finditer(text):
        before = text[start : m.start() + 1]
        last_word_match = re.search(r"(\S+)$", before)
        token = (
            last_word_match.group(1).lstrip("\"'(\u201c\u2018[").rstrip(".").lower()
            if last_word_match
            else ""
        )
        # Skip abbreviations and single initials ("J. Smith").
        if text[m.start()] == "." and (token in _ABBREVIATIONS or re.fullmatch(r"[a-z]", token)):
            continue
        sentences.append(text[start : m.end()].strip())
        start = m.end()
    sentences.append(text[start:].strip())
    return [s for s in sentences if s]


def _normalise_for_match(s: str) -> str:
    """Lowercase and collapse whitespace for sentence matching."""
    return re.sub(r"\s+", " ", s).strip().lower()


def _wrap_quadro(label: str, caption: str, lines: list[str], source: str) -> str:
    """Wrap lines in a single-column quadro environment."""
    body = "\\\\\n".join(lines)
    return (
        "\\begin{quadro}[htbp]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\footnotesize\n"
        "\\begin{tabular}{|p{0.93\\textwidth}|}\n\\hline\n"
        f"{body}\\\\\n\\hline\n"
        f"\\end{{tabular}}\n\\legend{{{source}}}\n\\end{{quadro}}\n"
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_llm_summaries(project_root: Path) -> dict[str, dict[str, str]]:
    """Load per-provider LLM-generated summaries from data/summaries_<provider>/."""
    summaries: dict[str, dict[str, str]] = defaultdict(dict)
    for provider in PROVIDERS:
        folder = summaries_dir(provider)
        if not folder.is_dir():
            log.warning("Summaries directory not found: %s", folder)
            continue
        for txt_file in folder.glob("*.txt"):
            summaries[provider][txt_file.stem] = txt_file.read_text(encoding="utf-8").strip()
    return summaries


def load_annotations(uid: str) -> list[list[str]]:
    """Load the 3 annotators' category picks for a single article.

    Returns [[class1, class2, class3], ...] for each annotator.
    """
    df = pd.read_excel(XLSX_PATH, sheet_name="alldiv")
    # Find the row matching this uid.
    title_col = "News"
    matched = df[
        df[title_col]
        .astype(str)
        .str.replace(r"\W", "", regex=True)
        .str.lower()
        .str.contains(uid, regex=False)
    ]
    if matched.empty:
        raise ValueError(f"Article {uid!r} not found in {XLSX_PATH}")

    row = matched.iloc[0]
    result: list[list[str]] = []
    for annotator in ANNOTATOR_NAMES:
        cols = ANNOTATOR_COLS[annotator]
        picks = []
        for col in cols:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                label = canonicalise_label(str(val))
                picks.append(label if label else str(val).strip())
        result.append(picks)
    return result


def load_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load single-label and percent-distribution prediction CSVs."""
    single_frames, percent_frames = [], []
    for provider in PROVIDERS:
        df_s = pd.read_csv(results_single_path(provider))
        df_p = pd.read_csv(results_percent_path(provider))
        for df in (df_s, df_p):
            df["provider"] = provider
            # Map provider-specific summary to generic "own_summary".
            df["input_type"] = df["input_type"].replace({f"summary_{provider}": "own_summary"})
        single_frames.append(df_s)
        percent_frames.append(df_p)
    return pd.concat(single_frames, ignore_index=True), pd.concat(percent_frames, ignore_index=True)


def percent_top1(row: pd.Series) -> tuple[list[str], dict[str, float]]:
    """Extract top-1 label(s) and full distribution from a percent row."""
    dist = {c: float(row[c]) for c in LABELS_CNN18 if pd.notna(row.get(c)) and float(row[c]) > 0}
    if not dist:
        return [], {}
    max_val = max(dist.values())
    return sorted(c for c, v in dist.items() if v == max_val), dist


# ---------------------------------------------------------------------------
# LaTeX generators
# ---------------------------------------------------------------------------


def _score_summary(annotations: list[list[str]]) -> list[tuple[str, int]]:
    """Compute weighted scores from raw annotations (reuses answer_key logic)."""
    row_data = {}
    for i, annotator in enumerate(ANNOTATOR_NAMES):
        cols = ANNOTATOR_COLS[annotator]
        for j, col in enumerate(cols):
            if j < len(annotations[i]):
                row_data[col] = annotations[i][j]
    row = pd.Series(row_data)
    ranking = compute_ranking(row)
    return sorted(ranking.items(), key=lambda x: (-x[1], x[0]))


def _build_answer_key_table(
    annotations: list[list[str]],
    title: str,
    label: str,
    position: str,
) -> str:
    """Build a LaTeX quadro showing annotator picks and weighted answer key."""
    rows_tex = ""
    for i, picks in enumerate(annotations, start=1):
        cells = (picks + ["---"] * 3)[:3]
        rows_tex += f"Anotador {i} & " + " & ".join(cells) + " \\\\\n\\hline\n"

    scores = _score_summary(annotations)
    max_pts = scores[0][1] if scores else 0
    score_str = ", ".join(f"{cat} ({pts})" for cat, pts in scores)
    top1_str = " / ".join(cat for cat, pts in scores if pts == max_pts)

    return (
        f"\\begin{{quadro}}[{position}]\n\\centering\n"
        f"\\caption{{Anotações humanas e gabarito ponderado da notícia {title}}}\n"
        f"\\label{{{label}}}\n\\footnotesize\n"
        "\\begin{tabular}{|l|c|c|c|}\n\\hline\n"
        "\\textbf{Anotador} & \\textbf{1\\textsuperscript{a} (3 pts)} "
        "& \\textbf{2\\textsuperscript{a} (2 pts)} "
        "& \\textbf{3\\textsuperscript{a} (1 pt)} \\\\\n\\hline\n"
        + rows_tex
        + f"\\multicolumn{{4}}{{|p{{0.85\\textwidth}}|}}"
        f"{{\\textbf{{Resultado:}} {score_str} $\\Rightarrow$ "
        f"\\textit{{top-1}} = {top1_str}}} \\\\\n\\hline\n"
        "\\end{tabular}\n"
        "\\legend{Fonte: elaborado pela autora (2026), "
        "a partir das anotações do CNN-Corpus.}\n"
        "\\end{quadro}\n"
    )


def _build_prediction_tables(
    uid: str,
    df_single: pd.DataFrame,
    df_percent: pd.DataFrame,
    title: str,
    label_suffix: str = "ref",
    position: str = "htbp",
) -> str:
    """Build single-label and percent top-1 prediction tables."""
    s = df_single[df_single.uid == uid]
    p = df_percent[df_percent.uid == uid]

    header = " & ".join(f"\\textbf{{{INPUT_DISPLAY[inp]}}}" for inp in INPUT_TYPES)

    def _make_rows(method: str) -> str:
        lines = ""
        for provider in PROVIDERS:
            vals = []
            for inp in INPUT_TYPES:
                if method == "single":
                    row = s[(s.provider == provider) & (s.input_type == inp)]
                    if len(row) and pd.notna(row.predicted_label.iloc[0]):
                        vals.append(escape_tex(str(row.predicted_label.iloc[0])))
                    else:
                        vals.append("invalid")
                else:
                    row = p[(p.provider == provider) & (p.input_type == inp)]
                    if len(row):
                        tops, _ = percent_top1(row.iloc[0])
                        vals.append(" / ".join(tops) if tops else "invalid")
                    else:
                        vals.append("---")
            lines += f"{PROVIDER_DISPLAY[provider]} & " + " & ".join(vals) + " \\\\\n"
        return lines

    def _wrap_table(caption: str, label: str, rows: str, note: str = "") -> str:
        return (
            f"\\begin{{table}}[{position}]\n\\centering\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\footnotesize\n"
            "\\begin{tabular}"
            "{p{0.12\\textwidth}p{0.18\\textwidth}p{0.18\\textwidth}"
            "p{0.18\\textwidth}p{0.18\\textwidth}}\n\\hline\n"
            f"\\textbf{{Modelo}} & {header} \\\\\n\\hline\n"
            f"{rows}\\hline\n\\end{{tabular}}\n"
            "\\legend{Fonte: elaborada pela autora (2026).}\n"
            + (f"\\legend{{{note}}}\n" if note else "")
            + "\\end{table}\n"
        )

    single_tex = _wrap_table(
        f"Rótulos atribuídos no método single-label à notícia {title}",
        f"tab:{label_suffix}-single",
        _make_rows("single"),
    )
    percent_tex = _wrap_table(
        f"Categoria de maior percentual no método percent-distribution "
        f"para a notícia {title}",
        f"tab:{label_suffix}-percent",
        _make_rows("percent"),
        "Nota: ``A / B'' indica empate no percentual máximo.",
    )
    return single_tex + "\n" + percent_tex


# ---------------------------------------------------------------------------
# Main generators
# ---------------------------------------------------------------------------


def generate_reference_article(
    article: Article,
    summaries: dict[str, dict[str, str]],
    annotations: list[list[str]],
    df_single: pd.DataFrame,
    df_percent: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate LaTeX files for the reference article (ref)."""
    examples_dir = output_dir / "chapters" / "metodologia" / "exemplos"
    examples_dir.mkdir(parents=True, exist_ok=True)

    title_tex = f"``{escape_tex(article.title)}''"
    sentences = split_sentences(article.full_text)

    sent_index = {_normalise_for_match(s): i for i, s in enumerate(sentences, start=1)}

    # Full text with numbered sentences.
    fulltext_lines = [f"{i}. {escape_tex(s)}" for i, s in enumerate(sentences, 1)]
    (examples_dir / "ref_fulltext.tex").write_text(
        _wrap_quadro(
            "qua:ref-fulltext",
            f"Texto completo da notícia {title_tex}, com sentenças numeradas",
            fulltext_lines,
            "Fonte: CNN-Corpus \\cite{10.1145/3342558.3345388}.",
        ),
        encoding="utf-8",
    )

    # Gold standard (extractive summary).
    gold_lines = [
        f"{sent_index.get(_normalise_for_match(s), '--')}. {escape_tex(s)}"
        for s in split_sentences(article.gold_standard)
    ]
    (examples_dir / "ref_goldstandard.tex").write_text(
        _wrap_quadro(
            "qua:ref-gold",
            f"Resumo extrativo de referência (\\textit{{gold standard}}) "
            f"da notícia {title_tex}",
            gold_lines,
            "Fonte: CNN-Corpus \\cite{10.1145/3342558.3345388}.",
        ),
        encoding="utf-8",
    )

    # Highlights.
    hl_lines = [escape_tex(s) for s in split_sentences(article.highlights)]
    (examples_dir / "ref_highlights.tex").write_text(
        _wrap_quadro(
            "qua:ref-highlights",
            f"\\textit{{Highlights}} editoriais da notícia {title_tex}",
            hl_lines,
            "Fonte: CNN-Corpus \\cite{10.1145/3342558.3345388}.",
        ),
        encoding="utf-8",
    )

    # LLM summaries.
    summary_rows = ""
    for provider in PROVIDERS:
        text = summaries[provider].get(article.uid, "(summary not found)")
        summary_rows += f"{PROVIDER_DISPLAY[provider]} & {escape_tex(text)} \\\\\n\\hline\n"
    (examples_dir / "ref_llm_summaries.tex").write_text(
        "\\begin{quadro}[htbp]\n\\centering\n"
        f"\\caption{{Resumos gerados pelos três modelos para a notícia {title_tex}}}\n"
        "\\label{qua:ref-llm-summaries}\n\\footnotesize\n"
        "\\begin{tabular}{|p{0.13\\textwidth}|p{0.77\\textwidth}|}\n\\hline\n"
        "\\textbf{Modelo} & \\textbf{Resumo gerado} \\\\\n\\hline\n"
        + summary_rows
        + "\\end{tabular}\n"
        "\\legend{Fonte: elaborado pela autora (2026).}\n"
        "\\end{quadro}\n",
        encoding="utf-8",
    )

    # Answer key.
    (examples_dir / "ref_answer_key.tex").write_text(
        _build_answer_key_table(annotations, title_tex, "qua:ref-answer-key", "htbp"),
        encoding="utf-8",
    )

    # Verify predictions (ref_predicoes.tex is hand-written; just log for
    # manual comparison).
    log.info("Prediction check for ref (compare with existing .tex):")
    for provider in PROVIDERS:
        for inp in INPUT_TYPES:
            rs = df_single[
                (df_single.uid == article.uid)
                & (df_single.provider == provider)
                & (df_single.input_type == inp)
            ]
            rp = df_percent[
                (df_percent.uid == article.uid)
                & (df_percent.provider == provider)
                & (df_percent.input_type == inp)
            ]
            label = rs.predicted_label.iloc[0] if len(rs) else None
            dist = percent_top1(rp.iloc[0])[1] if len(rp) else {}
            log.info(
                "  %-8s %-22s single=%-25s percent=%s",
                provider,
                inp,
                label,
                dist,
            )


def generate_appendix(
    corpus: list[Article],
    summaries: dict[str, dict[str, str]],
    df_single: pd.DataFrame,
    df_percent: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate the per-category appendix (one example article per category)."""
    uid_to_article = {a.uid: a for a in corpus}

    parts: list[str] = [
        "% Appendix C --- generated by scripts/generate_thesis_examples.py\n"
        "\\chapter{Exemplos de notícias e classificações por categoria}\n"
        "\\label{apx:examples}\n\n"
        "Este apêndice apresenta um exemplo de notícia de cada uma das 18 "
        "categorias do CNN-Corpus. Os exemplos são os mesmos textos apresentados "
        "nos Apêndices A a R de \\citeonline{silva2020sumarizacao}, o que permite "
        "ao leitor comparar os dois trabalhos. Para cada notícia, apresentam-se "
        "o texto completo, com sentenças numeradas; o resumo extrativo de "
        "referência, com o número que cada sentença ocupa no texto "
        "completo; os \\textit{highlights}; os resumos gerados pelos três "
        "modelos; as anotações humanas e o gabarito ponderado; e os "
        "rótulos atribuídos pelos modelos nos dois métodos de "
        "classificação. A categoria que dá nome a cada seção "
        "é a usada por \\citeonline{silva2020sumarizacao} para o primeiro "
        "nível do corpus, que pode diferir do \\textit{top-1} do gabarito "
        "ponderado desta dissertação.\n\n"
    ]

    for category, uid in EXAMPLES_BY_CATEGORY.items():
        article = uid_to_article.get(uid)
        if article is None:
            log.warning("Article %s not found in corpus, skipping.", uid)
            continue

        slug = re.sub(r"[^a-z]", "", category.lower())
        title_tex = (
            f"``{escape_tex(article.title)}''"
            if getattr(article, "title", "")
            else f"\\texttt{{{uid}}}"
        )
        sentences = split_sentences(article.full_text)
        sent_index = {_normalise_for_match(s): i for i, s in enumerate(sentences, start=1)}

        # Section header.
        parts.append(f"\\section{{{category}}}\n\\label{{apx:examples-{slug}}}\n\n")

        # Full text.
        numbered = "\n\n".join(
            f"\\noindent {i}. {escape_tex(s)}" for i, s in enumerate(sentences, 1)
        )
        parts.append(
            f"\\noindent\\textbf{{Título:}} {title_tex}\n\n"
            "\\medskip\\noindent\\textbf{Texto completo}\n\n"
            f"{{\\footnotesize\n{numbered}\n\\par}}\n\n"
        )

        # Extractive summary.
        gold_sentences = split_sentences(article.gold_standard)
        gold_tex = "\n\n".join(
            f"\\noindent {sent_index.get(_normalise_for_match(s), '--')}. {escape_tex(s)}"
            for s in gold_sentences
        )
        parts.append(
            "\\medskip\\noindent\\textbf{Resumo extrativo de "
            f"referência}}\n\n{{\\footnotesize\n{gold_tex}\n\\par}}\n\n"
        )

        # Highlights.
        hl_tex = "\n\n".join(
            f"\\noindent {escape_tex(s)}" for s in split_sentences(article.highlights)
        )
        parts.append(
            "\\medskip\\noindent\\textbf{\\textit{Highlights}}\n\n"
            f"{{\\footnotesize\n{hl_tex}\n\\par}}\n\n"
        )

        # LLM-generated summaries.
        for provider in PROVIDERS:
            text = summaries[provider].get(uid, "(summary not found)")
            parts.append(
                f"\\medskip\\noindent\\textbf{{Resumo gerado pelo "
                f"{PROVIDER_DISPLAY[provider]}}}\n\n"
                f"{{\\footnotesize\\noindent {escape_tex(text)}\\par}}\n\n"
            )

        # Answer key table.
        try:
            annotations = load_annotations(uid)
            parts.append(
                _build_answer_key_table(
                    annotations,
                    title_tex,
                    f"qua:apx-{slug}-answer-key",
                    "H",
                )
                + "\n"
            )
        except ValueError as exc:
            log.warning("(%s): %s", uid, exc)

        # Prediction tables.
        parts.append(
            _build_prediction_tables(
                uid,
                df_single,
                df_percent,
                title_tex,
                label_suffix=f"apx-{slug}",
                position="H",
            )
            + "\n\\clearpage\n\n"
        )

    appendix_dir = output_dir / "appendix"
    appendix_dir.mkdir(parents=True, exist_ok=True)
    (appendix_dir / "exemplos_por_classe.tex").write_text("".join(parts), encoding="utf-8")
    log.info(
        "Appendix written: %s (%d categories)",
        appendix_dir / "exemplos_por_classe.tex",
        len(EXAMPLES_BY_CATEGORY),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: load data, generate all LaTeX files."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Generate LaTeX thesis example snippets from project data."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Thesis output directory (e.g. ../dissertacao_v2)",
    )
    args = parser.parse_args()

    log.info("Loading corpus...")
    corpus = load_corpus()
    uid_to_article = {a.uid: a for a in corpus}

    log.info("Loading LLM summaries...")
    summaries = load_llm_summaries(Path("."))

    log.info("Loading predictions...")
    df_single, df_percent = load_predictions()

    log.info("Loading annotations for reference article...")
    annotations = load_annotations(REFERENCE_UID)

    ref_article = uid_to_article.get(REFERENCE_UID)
    if ref_article is None:
        log.error("Reference article %s not found in corpus.", REFERENCE_UID)
        return

    log.info("Generating reference article files...")
    generate_reference_article(
        ref_article, summaries, annotations, df_single, df_percent, args.output
    )

    log.info("Generating per-category appendix...")
    generate_appendix(corpus, summaries, df_single, df_percent, args.output)

    log.info("Done. Review generated files before compiling.")


if __name__ == "__main__":
    main()
