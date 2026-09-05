"""Plain-Python helpers (no decorators).

DSPy compilation, token counting, prose rendering, and the actual naive /
optimized prompting loops. Reasoners orchestrate; helpers do the work.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Optional

# DSPy imports are lazy — heavy module, only loaded when an optimizer runs.
# Each function that needs DSPy imports it locally.


# ---------- Constants ----------

DEFAULT_TASK = "Answer a factual question grounded in a short reading passage."


def naive_prompt(task_description: Optional[str] = None) -> str:
    """Hand-crafted prompt template. `{passage}` / `{question}` are filled per example.

    `task_description` is the one sentence both the naive run and DSPy start from.
    """
    task = (task_description or "").strip() or DEFAULT_TASK
    return (
        "You are a careful assistant.\n\n"
        f"Task: {task}\n\n"
        "Read the passage below, then answer the question using ONLY information from the passage.\n"
        'If the answer is not in the passage, say "I cannot determine the answer from the passage."\n\n'
        "Passage:\n{passage}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )


# Backward-compatible default used by fallbacks and older call sites.
NAIVE_RAW_PROMPT = naive_prompt()


# ---------- Fallback constructors (safe-default Pydantic instances) ----------

def fallback_example_result(index: int, expected: str) -> dict:
    """Return a safe-default per-example result when scoring fails."""
    return {
        "example_index": index,
        "predicted_answer": "[SCORING_FAILED]",
        "expected_answer": expected,
        "correct": False,
        "confidence": 0.0,
        "latency_ms": 0.0,
        "prompt_used": "",
        "token_count": 0,
        "confident": False,
        "reasoning": "Scoring failed; recorded as incorrect.",
    }


def fallback_strategy_result(strategy_name: str, optimizer: str) -> dict:
    return {
        "strategy_name": strategy_name,
        "accuracy": 0.0,
        "avg_latency_ms": 0.0,
        "total_tokens": 0,
        "per_example": [],
        "compiled_prompt": "[STRATEGY_FAILED]",
        "raw_prompt": NAIVE_RAW_PROMPT,
        "optimizer": optimizer,
        "confident": False,
    }


# ---------- Token / latency utilities ----------

_word_re = re.compile(r"\S+")


def approx_token_count(text: str) -> int:
    """Cheap token estimate: ~1 token per 0.75 words. Good enough for chart bars."""
    if not text:
        return 0
    return max(1, int(len(_word_re.findall(text)) * 1.33))


# ---------- Splitting / sampling ----------

def train_test_split(
    examples: list[dict], train_ratio: float
) -> tuple[list[dict], list[dict]]:
    """Deterministic 70/30-style split. First N go to train, rest to eval."""
    n = len(examples)
    cut = max(1, int(n * train_ratio))
    train = examples[:cut]
    eval_set = examples[cut:] if cut < n else examples
    return train, eval_set


def to_dspy_examples(rows: list[dict]) -> list[Any]:
    """Convert raw {passage, question, expected_answer} dicts into dspy.Example."""
    import dspy  # local import

    out = []
    for r in rows:
        ex = dspy.Example(
            passage=r["passage"],
            question=r["question"],
            expected_answer=r["expected_answer"],
        ).with_inputs("passage", "question")
        out.append(ex)
    return out


# ---------- DSPy prompt-program wrappers ----------

def run_naive_one(
    *,
    passage: str,
    question: str,
    model: Optional[str],
    task_description: Optional[str] = None,
) -> tuple[str, str, int, float]:
    """Run the hand-crafted prompt via the LLM and return (answer, prompt, tokens, latency_ms)."""
    from . import llm  # local helper module (created in this package)

    prompt = naive_prompt(task_description).format(passage=passage, question=question)
    t0 = time.perf_counter()
    answer = llm.complete(prompt, model=model)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    tokens = approx_token_count(prompt) + approx_token_count(answer)
    return answer.strip(), prompt, tokens, latency_ms


def run_dspy_predict_one(
    *,
    compiled_program: Any,
    passage: str,
    question: str,
) -> tuple[str, int, float]:
    """Run a compiled dspy.Predict on one example."""
    t0 = time.perf_counter()
    pred = compiled_program(passage=passage, question=question)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    answer = getattr(pred, "answer", str(pred))
    tokens = approx_token_count(passage) + approx_token_count(question) + approx_token_count(answer)
    return answer.strip(), tokens, latency_ms


def run_dspy_cot_one(
    *,
    compiled_program: Any,
    passage: str,
    question: str,
) -> tuple[str, int, float, str]:
    """Run a compiled dspy.ChainOfThought on one example. Returns reasoning too."""
    t0 = time.perf_counter()
    pred = compiled_program(passage=passage, question=question)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    answer = getattr(pred, "answer", str(pred))
    reasoning = getattr(pred, "rationale", "")
    tokens = (
        approx_token_count(passage)
        + approx_token_count(question)
        + approx_token_count(answer)
        + approx_token_count(reasoning)
    )
    return answer.strip(), tokens, latency_ms, reasoning
