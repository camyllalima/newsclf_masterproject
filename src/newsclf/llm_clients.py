"""LLM clients with on-disk caching and exponential-backoff retry.

CHECK the current model names and SDK details in each provider's documentation;
they change often.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from newsclf.config import (
    API_DELAY_PER_PROVIDER,
    API_DELAY_SECONDS,
    CACHE_DIR,
    OPENROUTER_BASE_URL,
    OPENROUTER_PROVENANCE_LOG,
    OPENROUTER_PROVIDER_PREFS,
)


def _cache_path(model: str, prompt: str) -> Path:
    digest = hashlib.sha256(f"{model}|{prompt}".encode()).hexdigest()
    return CACHE_DIR / f"{digest}.json"



def with_retry(call: Callable[[], str], attempts: int = 8) -> str:
    """Retry a call with exponential backoff (1s, 2s, 4s, ..., up to 128s)."""
    for attempt in range(attempts):
        try:
            return call()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def call_chatgpt(prompt: str, model: str) -> str:
    """Classify with OpenAI's ChatGPT."""
    from openai import OpenAI

    client = OpenAI()  # reads OPENAI_API_KEY
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def call_claude(prompt: str, model: str) -> str:
    """Classify with Anthropic's Claude."""
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    message = client.messages.create(
        model=model,
        max_tokens=100,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


_provenance_lock = threading.Lock()


class _RateLimiter:
    """Enforce a minimum interval between API call starts across threads."""

    def __init__(self, min_interval: float) -> None:
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._last: float = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


_rate_limiters: dict[str, _RateLimiter] = {}
_rate_limiters_lock = threading.Lock()


def _log_provenance(provider_name: str, model: str) -> None:
    """Append one line to the provenance JSONL log (thread-safe)."""
    import datetime

    OPENROUTER_PROVENANCE_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        "model": model,
        "infra_provider": provider_name,
    }
    with _provenance_lock:
        with OPENROUTER_PROVENANCE_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def call_deepseek(prompt: str, model: str) -> str:
    """Classify with DeepSeek via OpenRouter (OpenAI-compatible).

    Sends provider preferences (order + allow_fallbacks) and logs
    which infra provider actually served the request.
    """
    import os

    from openai import OpenAI

    log = logging.getLogger(__name__)
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ["OPEN_ROUTER_API_KEY"],
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"provider": OPENROUTER_PROVIDER_PREFS},
    )

    # Extract infra provider from OpenRouter's response metadata.
    provider_name = "unknown"
    if hasattr(response, "model_extra") and response.model_extra:
        provider_name = response.model_extra.get("provider", "unknown")
    log.debug("OpenRouter infra provider: %s", provider_name)
    _log_provenance(provider_name, model)

    if not response.choices:
        log.warning("OpenRouter returned no choices (deepseek) — treating as empty.")
        return ""
    content = response.choices[0].message.content
    if content is None:
        log.warning("OpenRouter returned None content (deepseek) — treating as empty.")
        return ""
    return content


def call_gemini(prompt: str, model: str) -> str:
    """Classify with Google's Gemini via OpenRouter (OpenAI-compatible)."""
    import os

    from openai import OpenAI

    log = logging.getLogger(__name__)
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ["OPEN_ROUTER_API_KEY"],
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    # Log infra provider for auditability.
    provider_name = "unknown"
    if hasattr(response, "model_extra") and response.model_extra:
        provider_name = response.model_extra.get("provider", "unknown")
    log.debug("OpenRouter infra provider (gemini): %s", provider_name)
    _log_provenance(provider_name, model)

    if not response.choices:
        log.warning("OpenRouter returned no choices (gemini) — treating as empty.")
        return ""
    content = response.choices[0].message.content
    if content is None:
        log.warning("OpenRouter returned None content (gemini) — treating as empty.")
        return ""
    return content


PROVIDERS: dict[str, Callable[[str, str], str]] = {
    "chatgpt": call_chatgpt,
    "claude": call_claude,
    "deepseek": call_deepseek,
    "gemini": call_gemini,
}


def _get_rate_limiter(provider: str) -> _RateLimiter:
    """Return a per-provider rate limiter, creating it on first access."""
    with _rate_limiters_lock:
        if provider not in _rate_limiters:
            delay = API_DELAY_PER_PROVIDER.get(provider, API_DELAY_SECONDS)
            _rate_limiters[provider] = _RateLimiter(delay)
        return _rate_limiters[provider]


def classify(provider: str, model: str, prompt: str) -> str:
    """Classify a prompt through a provider, applying cache, retry, and rate-limit delay."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}. Choose from: {list(PROVIDERS)}")
    provider_call = PROVIDERS[provider]
    path = _cache_path(model, prompt)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["response"]
    _get_rate_limiter(provider).wait()
    response = with_retry(lambda: provider_call(prompt, model))
    CACHE_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps({"response": response}), encoding="utf-8")
    return response
