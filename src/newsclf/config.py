"""Project configuration: labels, models, file paths and column names."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

load_dotenv(PROJECT_ROOT / ".env")

# --- Paths ---

# Corpus data directory.
DB3_DIR = PROJECT_ROOT / "data" / "db3"

# Results directory and sub-directories.
RESULTS_DIR = PROJECT_ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
EXPERIMENTS_DIR = PREDICTIONS_DIR / "experiments"
EVALUATION_DIR = RESULTS_DIR / "evaluation"

# Ground-truth annotations.
ANSWER_KEY_DIR = PROJECT_ROOT / "data" / "answer_key"

# On-disk LLM response cache.
CACHE_DIR = PROJECT_ROOT / "cache"

# Output directory for figures/plots.
FIGURES_DIR = PROJECT_ROOT / "figures"

# Summaries output directory and consolidated CSV.
SUMMARIES_DIR = PROJECT_ROOT / "data" / "summaries_chatgpt"
SUMMARIES_CSV = PROJECT_ROOT / "data" / "summaries_chatgpt.csv"
SUMMARIES_DEEPSEEK_DIR = PROJECT_ROOT / "data" / "summaries_deepseek"
SUMMARIES_DEEPSEEK_CSV = PROJECT_ROOT / "data" / "summaries_deepseek.csv"
SUMMARIES_GEMINI_DIR = PROJECT_ROOT / "data" / "summaries_gemini"
SUMMARIES_GEMINI_CSV = PROJECT_ROOT / "data" / "summaries_gemini.csv"

# --- Labels ---

# Prediction target: the Level 1 (primary) label.
LABELS_CNN18: list[str] = [
    "Arts",
    "Automotive",
    "Business",
    "Computer and Internet",
    "Economy and Finance",
    "Education",
    "Employment and Work",
    "Entertainment",
    "Food and Beverage",
    "Games",
    "Government and Politics",
    "Health",
    "Law",
    "Science",
    "Society and Culture",
    "Sports",
    "Travel",
    "Weather",
]

# Brief description per label — used in prompts to reduce ambiguity.
LABEL_DESCRIPTIONS: dict[str, str] = {
    "Arts": "Visual arts, performing arts, museums, galleries, and art exhibitions.",
    "Automotive": "Cars, trucks, vehicle industry, car reviews, and motor shows.",
    "Business": "Companies, corporate news, startups, mergers, and management.",
    "Computer and Internet": "Computing, software, hardware, cybersecurity, and online services.",
    "Economy and Finance": "Markets, stocks, banking, fiscal policy, and economic indicators.",
    "Education": "Schools, universities, academic research, and education policy.",
    "Employment and Work": "Jobs, labor market, hiring, unemployment, and workplace issues.",
    "Entertainment": "Movies, TV shows, music, celebrities, and pop culture.",
    "Food and Beverage": "Cooking, restaurants, food industry, drinks, and nutrition.",
    "Games": "Video games, board games, gaming industry, and esports.",
    "Government and Politics": "Elections, legislation, political parties, and public policy.",
    "Health": "Medicine, diseases, public health, wellness, and healthcare systems.",
    "Law": "Courts, trials, legal rulings, crime, and law enforcement.",
    "Science": "Scientific discoveries, space, physics, biology, and research studies.",
    "Society and Culture": "Social issues, demographics, religion, traditions, and lifestyle.",
    "Sports": "Athletic competitions, teams, players, leagues, and sporting events.",
    "Travel": "Tourism, destinations, airlines, hotels, and travel guides.",
    "Weather": "Forecasts, storms, climate events, and meteorological phenomena.",
}

# --- Input types ---
INPUT_TYPES: dict[str, str] = {
    "full_text": "full_text",
    "extractive_summary": "gold_standard",
    "abstractive_summary": "highlights",
    "summary_chatgpt": "chatgpt_summary",
    "summary_deepseek": "deepseek_summary",
    "summary_gemini": "gemini_summary",
}

# --- Models ---

# Model names — CHECK the current names in each provider's documentation.
MODELS: dict[str, str] = {
    "chatgpt": "gpt-4o-mini",
    "gemini": "google/gemini-3.5-flash-lite",
    "deepseek": "deepseek/deepseek-v4-flash-0731",
}

# OpenRouter base URL (used by deepseek provider).
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Preferred infra provider on OpenRouter for reproducibility.
# OpenInference is the most reliably available host for DeepSeek V4 Flash
# on OpenRouter (verified 2026-09-08). Together is a secondary option.
OPENROUTER_PROVIDER_PREFS = {
    "order": ["OpenInference"],
    "allow_fallbacks": True,
}

# Log file tracking which infra provider served each DeepSeek call.
OPENROUTER_PROVENANCE_LOG = RESULTS_DIR / "deepseek_provenance.jsonl"

# Pause between real API calls (cache miss only), per provider.
# Gemini free tier ~15 RPM → 4s; ChatGPT paid tier → 0.5s.
API_DELAY_PER_PROVIDER: dict[str, float] = {
    "gemini": 1.0,
    "chatgpt": 0.5,
    "claude": 1.0,
    "deepseek": 1.0,
}

API_DELAY_SECONDS: float = 4.0  # fallback default


# --- Summarisation ---

SUMMARY_PROMPT_TEMPLATE = """\
You are an expert news summarizer. Read the article below and write a \
concise abstractive summary in 3 to 4 sentences. The summary must capture \
the main points of the article. Do NOT copy sentences verbatim — rephrase \
in your own words. Output ONLY the summary, with no headings, labels, or \
extra text.

Article:
\"\"\"{text}\"\"\""""


_SUMMARIES_DIRS: dict[str, Path] = {
    "chatgpt": SUMMARIES_DIR,
    "deepseek": SUMMARIES_DEEPSEEK_DIR,
    "gemini": SUMMARIES_GEMINI_DIR,
}

_SUMMARIES_CSVS: dict[str, Path] = {
    "chatgpt": SUMMARIES_CSV,
    "deepseek": SUMMARIES_DEEPSEEK_CSV,
    "gemini": SUMMARIES_GEMINI_CSV,
}


def summaries_dir(provider: str) -> Path:
    """Return the per-provider summaries directory."""
    return _SUMMARIES_DIRS[provider]


def summaries_csv(provider: str) -> Path:
    """Return the per-provider summaries consolidated CSV path."""
    return _SUMMARIES_CSVS[provider]


def _provider_dir(provider: str) -> Path:
    """Return the per-provider results sub-directory, creating it if needed."""
    d = PREDICTIONS_DIR / provider
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_single_path(provider: str) -> Path:
    """Return the per-provider single-label results CSV path."""
    return _provider_dir(provider) / f"results_single_{provider}.csv"


def results_percent_path(provider: str) -> Path:
    """Return the per-provider percent results CSV path."""
    return _provider_dir(provider) / f"results_percent_{provider}.csv"
