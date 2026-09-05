"""Provider-aware LLM client for helpers and reasoners.

Supports two backends, selected per request:

  * **ollama** — native Ollama (`/api/chat`) with OpenAI-compat (`/v1/chat/completions`)
    tried first. Env: ``OLLAMA_BASE_URL``, ``OLLAMA_API_KEY``, ``OLLAMA_MODEL``.
  * **openai** — any OpenAI-compatible ``/v1/chat/completions`` server (OpenAI,
    OpenRouter, Groq, vLLM, LM Studio, Together, Fireworks, DeepSeek, xAI, …).
    Env: ``OPENAI_BASE_URL``, ``OPENAI_API_KEY`` / ``OPENROUTER_API_KEY``,
    ``OPENAI_MODEL``.

Selection (``LLM_PROVIDER=auto`` by default):

  1. Explicit ``LLM_PROVIDER=ollama|openai`` wins.
  2. Model prefix wins: ``ollama/…``, ``openai/…``, ``openrouter/…``, ``groq/…``, …
  3. Ollama-style tags (``gemma4:31b-cloud``) → ollama.
  4. ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` / ``OPENROUTER_API_KEY`` → openai.
  5. Otherwise ollama (backward compatible).

The same resolved config is used by ``complete()`` and by DSPy compilation.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

import httpx


# Match <thinking>...</thinking> blocks (Qwen, DeepSeek-R1) and freeform
# "Thinking Process:\n..." blocks (Qwen MLX) — both leak into the answer.
_THINKING_BLOCK_RE = re.compile(
    r"<thinking>.*?</thinking>",
    re.IGNORECASE | re.DOTALL,
)
_THINKING_PROCESS_RE = re.compile(
    r"Thinking\s+Process:\s*\n.*?(?=\n\s*\n|\Z)",
    re.IGNORECASE | re.DOTALL,
)

# First path segment → (logical provider, default OpenAI-compat base).
# Logical provider is "ollama" or "openai" (openai = any /v1/chat/completions).
_VENDOR_PREFIXES: dict[str, tuple[str, str]] = {
    "ollama": ("ollama", ""),
    "openai": ("openai", "https://api.openai.com/v1"),
    "openrouter": ("openai", "https://openrouter.ai/api/v1"),
    "groq": ("openai", "https://api.groq.com/openai/v1"),
    "together": ("openai", "https://api.together.xyz/v1"),
    "fireworks": ("openai", "https://api.fireworks.ai/inference/v1"),
    "deepseek": ("openai", "https://api.deepseek.com/v1"),
    "xai": ("openai", "https://api.x.ai/v1"),
    "mistral": ("openai", "https://api.mistral.ai/v1"),
}

_VENDOR_KEY_ENV: dict[str, tuple[str, ...]] = {
    "ollama": ("OLLAMA_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY", "OPENAI_API_KEY"),
    "groq": ("GROQ_API_KEY", "OPENAI_API_KEY"),
    "together": ("TOGETHER_API_KEY", "OPENAI_API_KEY"),
    "fireworks": ("FIREWORKS_API_KEY", "OPENAI_API_KEY"),
    "deepseek": ("DEEPSEEK_API_KEY", "OPENAI_API_KEY"),
    "xai": ("XAI_API_KEY", "OPENAI_API_KEY"),
    "mistral": ("MISTRAL_API_KEY", "OPENAI_API_KEY"),
}

_OPENAI_ALIASES = {"openai", "openai_compat", "openai-compatible", "openai_compatible"}


def _strip_thinking(text: str) -> str:
    if not text:
        return text
    text = _THINKING_BLOCK_RE.sub("", text)
    text = _THINKING_PROCESS_RE.sub("", text)
    return text.strip()


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _normalize_openai_base(url: str) -> str:
    """Ensure an OpenAI-compat base ends with /v1 and has no trailing slash."""
    base = (url or "").strip().rstrip("/")
    if not base:
        return "https://api.openai.com/v1"
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def _split_vendor_prefix(model: str) -> tuple[Optional[str], str]:
    """Return (vendor, remainder) when model starts with a known vendor prefix.

    ``openai/`` (empty remainder) is valid — it selects the provider and lets
    ``OPENAI_MODEL`` / ``OLLAMA_MODEL`` fill in the id.
    """
    if not model or "/" not in model:
        return None, model
    vendor, remainder = model.split("/", 1)
    vendor_l = vendor.lower()
    if vendor_l in _VENDOR_PREFIXES:
        return vendor_l, remainder
    return None, model


def _first_env(names: tuple[str, ...]) -> str:
    for name in names:
        value = _env(name)
        if value:
            return value
    return ""


def _message_content(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text") or part.get("content") or "")
            else:
                parts.append(str(part))
        content = "".join(parts)
    if content:
        return str(content)
    # Some reasoning models put the final answer in a sibling field.
    for key in ("reasoning_content", "thinking"):
        extra = message.get(key)
        if extra:
            return str(extra)
    return ""


@dataclass(frozen=True)
class LLMConfig:
    """Resolved provider + endpoint used by complete() and DSPy."""

    provider: str  # "ollama" | "openai"
    vendor: str  # original prefix vendor, e.g. "openrouter" or "ollama"
    api_model: str  # value sent in the JSON "model" field
    host: str  # origin used for Ollama native routes
    openai_base: str  # always …/v1
    api_key: str
    source_model: str  # the string the caller asked for (may include prefix)

    @property
    def chat_url(self) -> str:
        return f"{self.openai_base}/chat/completions"

    @property
    def models_url(self) -> str:
        return f"{self.openai_base}/models"

    @property
    def display_model(self) -> str:
        if self.source_model:
            return self.source_model
        if self.vendor and self.vendor != self.provider:
            return f"{self.vendor}/{self.api_model}"
        return f"{self.provider}/{self.api_model}"

    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if "openrouter.ai" in self.openai_base:
            headers["HTTP-Referer"] = _env(
                "OPENROUTER_HTTP_REFERER",
                "https://github.com/agentfield/dspy-benchmark-studio",
            )
            headers["X-Title"] = _env("OPENROUTER_APP_TITLE", "dspy-benchmark-studio")
        return headers

    def dspy_kwargs(self, max_tokens: int = 512) -> dict:
        """LiteLLM-style kwargs so DSPy talks to the same endpoint as complete()."""
        return {
            "model": f"openai/{self.api_model}",
            "api_base": self.openai_base,
            "api_key": self.api_key or "not-needed",
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }


def resolve_config(model: Optional[str] = None) -> LLMConfig:
    """Resolve provider, model id, base URL, and key from args + env."""
    requested = (model or "").strip() or _env("AI_MODEL")
    vendor, remainder = _split_vendor_prefix(requested)

    explicit = _env("LLM_PROVIDER", "auto").lower()
    if explicit in _OPENAI_ALIASES:
        explicit = "openai"
    if explicit not in ("auto", "ollama", "openai"):
        explicit = "auto"

    if explicit in ("ollama", "openai"):
        provider = explicit
    elif vendor:
        provider = _VENDOR_PREFIXES[vendor][0]
    elif requested and ":" in requested and "/" not in requested:
        provider = "ollama"
    elif _env("OPENAI_BASE_URL") or _env("OPENAI_API_KEY") or _env("OPENROUTER_API_KEY"):
        provider = "openai"
        if not vendor and _env("OPENROUTER_API_KEY") and not _env("OPENAI_API_KEY"):
            vendor = "openrouter"
    else:
        provider = "ollama"

    api_model = remainder if vendor else requested

    if provider == "ollama":
        vendor = "ollama"
    elif not vendor or _VENDOR_PREFIXES.get(vendor, ("openai",))[0] != "openai":
        vendor = "openai"

    if not api_model:
        if provider == "openai":
            api_model = _env("OPENAI_MODEL") or "gpt-4o-mini"
        else:
            api_model = _env("OLLAMA_MODEL") or "llama3.2"

    if provider == "ollama":
        host = _env("OLLAMA_BASE_URL") or "http://localhost:11434"
        host = host.rstrip("/")
        if host.endswith("/v1"):
            host = host[: -len("/v1")]
        openai_base = _normalize_openai_base(host)
        api_key = _first_env(_VENDOR_KEY_ENV["ollama"])
        return LLMConfig(
            provider="ollama",
            vendor=vendor,
            api_model=api_model,
            host=host,
            openai_base=openai_base,
            api_key=api_key,
            source_model=requested or api_model,
        )

    default_base = _VENDOR_PREFIXES.get(vendor, ("openai", "https://api.openai.com/v1"))[1]
    openai_base = _normalize_openai_base(
        _env("OPENAI_BASE_URL") or default_base or "https://api.openai.com/v1"
    )
    api_key = _first_env(_VENDOR_KEY_ENV.get(vendor, ("OPENAI_API_KEY",)))
    if not api_key:
        api_key = _env("OPENAI_API_KEY") or _env("OPENROUTER_API_KEY")
    # Host is the origin; native Ollama routes are not used in this branch.
    host = openai_base[: -len("/v1")] if openai_base.endswith("/v1") else openai_base
    return LLMConfig(
        provider="openai",
        vendor=vendor,
        api_model=api_model,
        host=host,
        openai_base=openai_base,
        api_key=api_key,
        source_model=requested or api_model,
    )


def list_available_models(config: Optional[LLMConfig] = None) -> list[str]:
    """Best-effort list of model IDs from the active provider."""
    cfg = config or resolve_config()
    out: list[str] = []
    try:
        with httpx.Client(timeout=2.0) as client:
            r = client.get(cfg.models_url, headers=cfg.headers())
            if r.status_code == 200:
                data = r.json()
                for m in data.get("data", []):
                    if isinstance(m, dict) and "id" in m:
                        out.append(m["id"])
                if out:
                    return out
            if cfg.provider == "ollama":
                r2 = client.get(f"{cfg.host}/api/tags", headers=cfg.headers())
                if r2.status_code == 200:
                    for m in r2.json().get("models", []):
                        if isinstance(m, dict) and "name" in m:
                            out.append(m["name"])
    except Exception:
        pass
    return out


def _complete_openai(client: httpx.Client, cfg: LLMConfig, prompt: str, max_tokens: int) -> str:
    payload = {
        "model": cfg.api_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    r = client.post(cfg.chat_url, json=payload, headers=cfg.headers())
    if r.status_code == 400 and "max_completion_tokens" in (r.text or ""):
        payload.pop("max_tokens", None)
        payload["max_completion_tokens"] = max_tokens
        r = client.post(cfg.chat_url, json=payload, headers=cfg.headers())
    if r.status_code != 200:
        raise RuntimeError(
            f"HTTP {r.status_code} from {cfg.chat_url} for model={cfg.api_model!r}: "
            f"{r.text[:300]}"
        )
    data = r.json()
    try:
        content = _message_content(data["choices"][0]["message"])
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(
            f"Unexpected /v1/chat/completions shape from {cfg.openai_base}: {data}"
        ) from e
    return _strip_thinking(content)


def _complete_ollama_native(client: httpx.Client, cfg: LLMConfig, prompt: str, max_tokens: int) -> str:
    r = client.post(
        f"{cfg.host}/api/chat",
        json={
            "model": cfg.api_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": max_tokens,
            },
        },
        headers=cfg.headers(),
    )
    if r.status_code != 200:
        raise RuntimeError(
            f"HTTP {r.status_code} from {cfg.host}/api/chat: {r.text[:200]}. "
            f"Available models: {list_available_models(cfg) or '<unknown>'}"
        )
    data = r.json()
    msg = data.get("message") or {}
    content = _message_content(msg) if isinstance(msg, dict) else ""
    return _strip_thinking(content)


def complete(
    prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 2048,
    config: Optional[LLMConfig] = None,
) -> str:
    """Call the resolved provider synchronously and return stripped text.

    Env (ollama):
      OLLAMA_BASE_URL  e.g. http://host.docker.internal:11434
      OLLAMA_API_KEY   bearer token if your Ollama is gated
      OLLAMA_MODEL     fallback model when none is passed

    Env (openai-compatible):
      OPENAI_BASE_URL  e.g. https://api.openai.com/v1  (or OpenRouter / vLLM / LM Studio)
      OPENAI_API_KEY   bearer token (or OPENROUTER_API_KEY)
      OPENAI_MODEL     fallback model when none is passed
      LLM_PROVIDER     auto | ollama | openai
    """
    cfg = config or resolve_config(model)
    last_err: Optional[Exception] = None

    with httpx.Client(timeout=180.0) as client:
        try:
            return _complete_openai(client, cfg, prompt, max_tokens)
        except httpx.HTTPError as e:
            last_err = e
        except RuntimeError as e:
            last_err = e
            # Ollama: 404 on the OpenAI-compat route is expected on older servers.
            if cfg.provider != "ollama":
                raise

        if cfg.provider == "ollama":
            try:
                return _complete_ollama_native(client, cfg, prompt, max_tokens)
            except httpx.HTTPError as e:
                last_err = e
            except RuntimeError as e:
                last_err = e

    raise RuntimeError(
        f"Could not reach {cfg.provider} for model={cfg.api_model!r} "
        f"at {cfg.openai_base if cfg.provider == 'openai' else cfg.host}. "
        f"Last error: {last_err}. "
        f"Available models: {list_available_models(cfg) or '<unknown>'}"
    )
