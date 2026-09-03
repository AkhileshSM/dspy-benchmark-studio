"""Thin LLM-call shim for the helpers module.

This isolates raw LLM calls so the helpers and reasoners don't need to know
which provider the agent is using.

Strategy:
  1. Try the OpenAI-compatible /v1/chat/completions endpoint (works on
     most cloud Ollama setups and on local Ollama >= 0.1.14).
  2. On 404, fall back to the native /api/chat endpoint with stream=false
     (more reliably supported across Ollama versions).
  3. Strip <thinking>...</thinking> blocks from the response — Qwen-style
     reasoning models emit these, and they pollute downstream parsing.
  4. On any error, raise with a clear message naming the model + URL tried.
"""

from __future__ import annotations

import os
import re
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


def _strip_thinking(text: str) -> str:
    if not text:
        return text
    text = _THINKING_BLOCK_RE.sub("", text)
    text = _THINKING_PROCESS_RE.sub("", text)
    return text.strip()


def _get_base_and_key() -> tuple[str, str]:
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    key = os.getenv("OLLAMA_API_KEY", "")
    return base, key


def list_available_models(base: Optional[str] = None) -> list[str]:
    """Best-effort list of model IDs available on the Ollama server."""
    base = base or _get_base_and_key()[0]
    out: list[str] = []
    try:
        with httpx.Client(timeout=10.0) as client:
            # OpenAI-compatible
            r = client.get(f"{base}/v1/models")
            if r.status_code == 200:
                data = r.json()
                for m in data.get("data", []):
                    if "id" in m:
                        out.append(m["id"])
                if out:
                    return out
            # Native Ollama
            r2 = client.get(f"{base}/api/tags")
            if r2.status_code == 200:
                for m in r2.json().get("models", []):
                    if "name" in m:
                        out.append(m["name"])
    except Exception:
        pass
    return out


def complete(prompt: str, model: Optional[str] = None, max_tokens: int = 2048) -> str:
    """Call a Cloud Ollama (or any OpenAI-compatible) endpoint synchronously.

    Env:
      OLLAMA_BASE_URL  e.g. http://host.docker.internal:11434
      OLLAMA_API_KEY   bearer token if your Ollama is gated
      OLLAMA_MODEL     fallback model when none is passed
    """
    base, key = _get_base_and_key()
    chosen_model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    last_err: Optional[Exception] = None

    with httpx.Client(timeout=180.0) as client:
        # ---- Attempt 1: OpenAI-compatible /v1/chat/completions ----
        try:
            r = client.post(
                f"{base}/v1/chat/completions",
                json={
                    "model": chosen_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": max_tokens,
                },
                headers=headers,
            )
            if r.status_code == 200:
                data = r.json()
                try:
                    content = data["choices"][0]["message"]["content"]
                    return _strip_thinking(content)
                except (KeyError, IndexError) as e:
                    raise RuntimeError(
                        f"Unexpected /v1/chat/completions shape from {base}: {data}"
                    ) from e
            # 404 → try the native endpoint
            if r.status_code == 404:
                last_err = RuntimeError(
                    f"404 from {base}/v1/chat/completions for model={chosen_model!r}. "
                    f"Available models: {list_available_models(base) or '<unknown>'}"
                )
            else:
                # Other non-200 — surface but still try the native endpoint as a fallback
                last_err = RuntimeError(
                    f"HTTP {r.status_code} from {base}/v1/chat/completions: {r.text[:200]}"
                )
        except httpx.HTTPError as e:
            last_err = e

        # ---- Attempt 2: Native Ollama /api/chat (stream=false) ----
        try:
            r2 = client.post(
                f"{base}/api/chat",
                json={
                    "model": chosen_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": max_tokens,
                    },
                },
                headers=headers,
            )
            if r2.status_code == 200:
                data = r2.json()
                msg = data.get("message") or {}
                content = msg.get("content", "")
                if not content and "thinking" in msg:
                    # Some Qwen variants put the final answer in a different field
                    content = msg.get("thinking", "")
                return _strip_thinking(content)
            last_err = RuntimeError(
                f"HTTP {r2.status_code} from {base}/api/chat: {r2.text[:200]}. "
                f"Available models: {list_available_models(base) or '<unknown>'}"
            )
        except httpx.HTTPError as e:
            last_err = e

    raise RuntimeError(
        f"Could not reach Ollama for model={chosen_model!r} at {base}. "
        f"Last error: {last_err}. "
        f"Available models: {list_available_models(base) or '<unknown>'}"
    )
