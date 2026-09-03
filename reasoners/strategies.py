"""Strategies router — the 3 Q&A strategies + single_example_runner.

Each strategy is itself a small orchestrator: it compiles the DSPy program
on the train split (synchronously, since DSPy program objects cannot
survive JSON serialization across app.call), then fans out across the
eval set via asyncio.gather over single_example_runner. The leaf runner
calls the shared confidence_scorer.

The compile step is wrapped in a helper that returns a JSON-serializable
"compiled state" — the *program* is rebuilt from the saved prompt at run
time by direct LLM completion with the compiled prompt as the system
message. This is faithful to what DSPy actually does (it rewrites the
prompt) and keeps the architecture honest about the cross-boundary
boundary.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional

from agentfield import AgentRouter

from . import helpers
from .models import (
    ConfidenceScore,
    ExampleResult,
    StrategyResult,
)

NODE_ID = os.getenv("AGENT_NODE_ID", "dspy-benchmark-studio")
router = AgentRouter(prefix="", tags=["strategy"])


# ---------------------------------------------------------------------------
# shared leaf: confidence_scorer
# ---------------------------------------------------------------------------

@router.reasoner()
async def confidence_scorer(
    predicted_answer: str,
    expected_answer: str,
    passage: str = "",
    question: str = "",
    model: Optional[str] = None,
) -> ConfidenceScore:
    """Score one prediction against ground truth.

    Hybrid: deterministic normalization + substring check first, then
    escalate to a fast LLM-as-judge only on disagreement with non-trivial
    expected. Keeps scoring cheap for easy wins; handles rephrasings.
    """
    from . import llm

    if not predicted_answer or not expected_answer:
        return ConfidenceScore(
            correct=False, confidence=0.0, confident=False, method="empty",
        )

    pred_norm = predicted_answer.strip().lower().rstrip(".!?:;,'\"")
    exp_norm = expected_answer.strip().lower().rstrip(".!?:;,'\"")

    if pred_norm == exp_norm or exp_norm in pred_norm or pred_norm in exp_norm:
        return ConfidenceScore(
            correct=True, confidence=0.99, confident=True, method="substring",
        )

    # Disagreement — escalate to LLM-as-judge
    judge_prompt = (
        "You are a strict but fair grader. Determine whether the predicted "
        "answer is semantically equivalent to the expected answer for the "
        "given question. Reply with JSON: {\"equivalent\": true|false, "
        "\"confidence\": 0.0-1.0}.\n\n"
        f"Passage: {passage[:600]}\n"
        f"Question: {question}\n"
        f"Expected: {expected_answer}\n"
        f"Predicted: {predicted_answer}\n"
    )
    try:
        raw = llm.complete(judge_prompt, model=model, max_tokens=512)
        import json
        import re
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            data = json.loads(m.group(0))
            return ConfidenceScore(
                correct=bool(data.get("equivalent", False)),
                confidence=float(data.get("confidence", 0.5)),
                confident=True,
                method="llm_judge",
            )
    except Exception:
        pass

    return ConfidenceScore(
        correct=False, confidence=0.4, confident=True, method="fallback_no_match",
    )


# ---------------------------------------------------------------------------
# atomic leaf: single_example_runner
# ---------------------------------------------------------------------------

@router.reasoner()
async def single_example_runner(
    strategy: str,
    example_index: int,
    passage: str,
    question: str,
    expected_answer: str,
    prompt_to_use: str = "",
    model: Optional[str] = None,
) -> ExampleResult:
    """Run ONE example through ONE strategy. Atomic leaf — no further sub-calls.

    For "naive": use the standard hand-crafted prompt.
    For "dspy_predict" / "dspy_cot": use prompt_to_use (the compiled prompt
    produced by the optimizer and passed in by the strategy reasoner).
    """
    try:
        prompt = prompt_to_use or helpers.NAIVE_RAW_PROMPT
        if strategy == "naive":
            prompt = helpers.NAIVE_RAW_PROMPT

        # All three strategies go through the same direct LLM completion path.
        # The difference is the prompt (raw vs compiled) and whether we ask
        # for chain-of-thought reasoning.
        from . import llm
        import time
        import json
        import re

        if strategy == "dspy_cot":
            user_prompt = (
                f"{prompt}\n\n---\n"
                f"Passage: {passage}\n"
                f"Question: {question}\n\n"
                f"Think step by step. First write a brief rationale, then on a "
                f"new line write 'Answer: <your answer>'."
            )
        else:
            user_prompt = (
                f"{prompt}\n\n---\n"
                f"Passage: {passage}\n"
                f"Question: {question}\n\n"
                f"Answer:"
            )

        t0 = time.perf_counter()
        raw = llm.complete(user_prompt, model=model, max_tokens=2048)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # llm.complete() already strips <thinking> blocks; do an extra
        # belt-and-braces pass here in case the model emits them inline.
        clean = re.sub(r"<thinking>.*?</thinking>", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
        if not clean:
            clean = raw.strip()

        # Parse answer + optional rationale
        answer = clean
        reasoning = ""
        if strategy == "dspy_cot":
            m = re.search(r"(?im)^answer\s*:\s*(.+)$", clean)
            if m:
                answer = m.group(1).strip()
                # Reasoning is everything before the Answer: line
                reasoning = clean[: m.start()].strip()
            else:
                # Last non-empty line fallback
                lines = [l for l in clean.splitlines() if l.strip()]
                if lines:
                    answer = lines[-1].strip()

        # Score via the shared leaf
        score_dict = await router.call(
            f"{NODE_ID}.confidence_scorer",
            predicted_answer=answer,
            expected_answer=expected_answer,
            passage=passage,
            question=question,
            model=model,
        )
        score = ConfidenceScore(**score_dict)

        tokens = (
            helpers.approx_token_count(user_prompt)
            + helpers.approx_token_count(clean)
        )
        return ExampleResult(
            example_index=example_index,
            predicted_answer=answer,
            expected_answer=expected_answer,
            correct=score.correct,
            confidence=score.confidence,
            latency_ms=latency_ms,
            prompt_used=prompt,
            token_count=tokens,
            confident=score.confident,
            reasoning=reasoning,
        )
    except Exception as e:
        return ExampleResult(
            example_index=example_index,
            predicted_answer=f"[RUNNER_ERROR:{type(e).__name__}]",
            expected_answer=expected_answer,
            correct=False, confidence=0.0,
            latency_ms=0.0, prompt_used="", token_count=0,
            confident=False, reasoning=str(e),
        )


# ---------------------------------------------------------------------------
# strategy reasoners (composers — compile, then fan out)
# ---------------------------------------------------------------------------

async def _run_strategy(
    strategy_name: str,
    optimizer_name: str,
    train_examples: list[dict],
    eval_examples: list[dict],
    model: Optional[str],
    enhanced: bool = False,
    max_bootstrapped_demos: int = 4,
    num_trials: int = 12,
) -> StrategyResult:
    """Shared body for naive / dspy_predict / dspy_cot strategies.

    For naive: skip compile, just fan out across eval set with the raw prompt.
    For dspy_*: compile the DSPy program on the train split (synchronously,
    since DSPy objects don't survive JSON serialization), capture the
    resulting compiled_prompt, then fan out using that prompt.
    """
    raw_prompt = helpers.NAIVE_RAW_PROMPT
    compiled_prompt = raw_prompt

    if strategy_name != "naive":
        try:
            compiled_prompt, optimizer_used = _compile_dspy_prompt(
                strategy=strategy_name,
                optimizer=optimizer_name,
                train_examples=train_examples,
                model=model,
                enhanced=enhanced,
                max_bootstrapped_demos=max_bootstrapped_demos,
                num_trials=num_trials,
            )
            optimizer_name = optimizer_used
        except Exception as e:
            # Optimization failure → fall back to raw prompt with a marker
            compiled_prompt = (
                raw_prompt
                + f"\n\n[NOTE: DSPy optimization failed: {e}. Using raw prompt.]"
            )
            optimizer_name = "failed"

    # Fan out across the eval set
    tasks = []
    for i, ex in enumerate(eval_examples):
        tasks.append(
            router.call(
                f"{NODE_ID}.single_example_runner",
                strategy=strategy_name,
                example_index=i,
                passage=ex["passage"],
                question=ex["question"],
                expected_answer=ex["expected_answer"],
                prompt_to_use=compiled_prompt,
                model=model,
            )
        )
    raw_results = await asyncio.gather(*tasks)
    per_example = [ExampleResult(**r) for r in raw_results]

    n = len(per_example) or 1
    correct = sum(1 for r in per_example if r.correct)
    accuracy = correct / n
    avg_latency = sum(r.latency_ms for r in per_example) / n
    total_tokens = sum(r.token_count for r in per_example)

    return StrategyResult(
        strategy_name=strategy_name,
        accuracy=accuracy,
        avg_latency_ms=avg_latency,
        total_tokens=total_tokens,
        per_example=per_example,
        compiled_prompt=compiled_prompt,
        raw_prompt=raw_prompt,
        optimizer=optimizer_name,
        confident=all(r.confident for r in per_example) if per_example else False,
    )


def _compile_dspy_prompt(
    *,
    strategy: str,
    optimizer: str,
    train_examples: list[dict],
    model: Optional[str],
    enhanced: bool,
    max_bootstrapped_demos: int,
    num_trials: int,
) -> tuple[str, str]:
    """Compile a DSPy program and return (compiled_prompt, optimizer_name).

    DSPy program objects don't survive JSON serialization across app.call,
    so the compilation happens here, in-process. The output is a string
    prompt that single_example_runner can use directly via the llm module.
    This is faithful to what DSPy actually does — it rewrites the prompt —
    and keeps the agent's DAG honest (you can see optimizer compile, then
    eval fan-out).
    """
    import dspy

    chosen_model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
    ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    ollama_key = os.getenv("OLLAMA_API_KEY", "")

    try:
        lm = dspy.LM(
            model=f"openai/{chosen_model}",
            api_base=f"{ollama_base}/v1",
            api_key=ollama_key or "ollama",
            max_tokens=512,
            temperature=0.0,
        )
        dspy.configure(lm=lm)
    except Exception:
        return helpers.NAIVE_RAW_PROMPT, "fallback"

    class QASignature(dspy.Signature):
        """Answer a factual question given a passage."""
        passage: str = dspy.InputField(desc="A short passage of text")
        question: str = dspy.InputField(desc="A factual question about the passage")
        answer: str = dspy.OutputField(desc="A concise answer grounded in the passage")

    if strategy == "dspy_predict":
        program = dspy.Predict(QASignature)
    else:  # dspy_cot
        program = dspy.ChainOfThought(QASignature)

    if not train_examples:
        return helpers.NAIVE_RAW_PROMPT, "no_trainset"

    trainset = helpers.to_dspy_examples(train_examples)

    # Compile
    compiled_program = program
    optimizer_used = optimizer
    try:
        if optimizer == "BootstrapFewShot":
            opt = dspy.BootstrapFewShot(
                metric=_metric,
                max_bootstrapped_demos=max_bootstrapped_demos,
            )
            compiled_program = opt.compile(program, trainset=trainset)
        elif optimizer == "MIPROv2":
            try:
                opt = dspy.MIPROv2(
                    metric=_metric,
                    auto="light",
                    num_trials=num_trials,
                )
                compiled_program = opt.compile(program, trainset=trainset)
            except Exception:
                # MIPROv2 fails on tiny trainsets — fall back
                opt = dspy.BootstrapFewShot(
                    metric=_metric,
                    max_bootstrapped_demos=max_bootstrapped_demos,
                )
                compiled_program = opt.compile(program, trainset=trainset)
                optimizer_used = "BootstrapFewShot"
    except Exception:
        return helpers.NAIVE_RAW_PROMPT, "compile_failed"

    return _render_compiled_prompt(compiled_program), optimizer_used


def _metric(example, pred, trace=None) -> float:
    """DSPy metric: substring match on the answer field."""
    try:
        gold = getattr(example, "expected_answer", "").strip().lower()
        pred_ans = getattr(pred, "answer", str(pred)).strip().lower()
        return float(gold in pred_ans or pred_ans in gold)
    except Exception:
        return 0.0


def _render_compiled_prompt(program: Any) -> str:
    """Render a compiled DSPy program into a representative prompt string.

    Used for the 'before/after' dashboard view. DSPy programs carry
    demonstrations + signature + instructions; we synthesize a single
    view that captures what the optimizer actually changed.
    """
    try:
        sig = None
        if hasattr(program, "signature"):
            sig = program.signature
        elif hasattr(program, "predict"):
            sig = program.predict.signature
        if sig is None:
            return helpers.NAIVE_RAW_PROMPT
        instructions = getattr(sig, "instructions", "") or "Answer the question using only the passage."
        # Collect demos (few-shot examples compiled by the optimizer)
        demos_section = ""
        demos_attr = getattr(program, "demos", None)
        if demos_attr:
            demo_list = list(demos_attr)
            if demo_list:
                lines = ["", "Compiled demonstrations (few-shot examples the optimizer picked):"]
                for d in demo_list[:3]:
                    p = (getattr(d, "passage", "") or "")[:200]
                    q = getattr(d, "question", "")
                    a = getattr(d, "answer", "")
                    lines.append(f"  Example passage: {p}...")
                    lines.append(f"  Example question: {q}")
                    lines.append(f"  Example answer: {a}")
                    lines.append("  ---")
                demos_section = "\n".join(lines)
        return (
            f"[DSPy-Compiled Prompt]\n\n"
            f"Task: {instructions}\n"
            f"Inputs: passage (a short passage of text), question (a factual question about the passage)\n"
            f"Output: answer (a concise answer grounded in the passage)\n"
            f"{demos_section}\n\n"
            f"Now answer the new question following the pattern above."
        )
    except Exception:
        return helpers.NAIVE_RAW_PROMPT


# ---------------------------------------------------------------------------
# entry points exposed to the orchestrator
# ---------------------------------------------------------------------------

@router.reasoner()
async def naive_prompting(
    train_examples: list[dict],
    eval_examples: list[dict],
    model: Optional[str] = None,
) -> StrategyResult:
    """Strategy 1: hand-crafted prompt, no DSPy, no optimization."""
    return await _run_strategy(
        strategy_name="naive",
        optimizer_name="none",
        train_examples=train_examples,
        eval_examples=eval_examples,
        model=model,
    )


@router.reasoner()
async def dspy_predict_optimized(
    train_examples: list[dict],
    eval_examples: list[dict],
    model: Optional[str] = None,
) -> StrategyResult:
    """Strategy 2: DSPy Predict compiled with BootstrapFewShot."""
    return await _run_strategy(
        strategy_name="dspy_predict",
        optimizer_name="BootstrapFewShot",
        train_examples=train_examples,
        eval_examples=eval_examples,
        model=model,
    )


@router.reasoner()
async def dspy_cot_optimized(
    train_examples: list[dict],
    eval_examples: list[dict],
    model: Optional[str] = None,
) -> StrategyResult:
    """Strategy 3: DSPy ChainOfThought compiled with MIPROv2."""
    return await _run_strategy(
        strategy_name="dspy_cot",
        optimizer_name="MIPROv2",
        train_examples=train_examples,
        eval_examples=eval_examples,
        model=model,
    )


@router.reasoner()
async def enhanced_cot_runner(
    eval_examples: list[dict],
    train_examples: list[dict],
    recommended_optimizer: str,
    max_bootstrapped_demos: int,
    num_trials: int,
    model: Optional[str] = None,
) -> StrategyResult:
    """The 'best-effort' retry round with stronger compilation settings.

    Invoked by the orchestrator when gap_analyzer says the initial round
    had a problem (accuracy < 0.3 or naive beat DSPy).
    """
    return await _run_strategy(
        strategy_name="dspy_cot",
        optimizer_name=recommended_optimizer,
        train_examples=train_examples,
        eval_examples=eval_examples,
        model=model,
        enhanced=True,
        max_bootstrapped_demos=max_bootstrapped_demos,
        num_trials=num_trials,
    )
