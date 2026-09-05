"""Meta router — orchestrator + gap_analyzer.

The dynamic orchestration lives here: the orchestrator reads
intermediate strategy scores, calls gap_analyzer, and conditionally
spawns an enhanced retry round.
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

from agentfield import AgentRouter

from . import helpers
from . import llm
from .models import (
    BenchmarkReport,
    EnhancedStrategyResult,
    GapAnalysis,
    StrategyResult,
)

NODE_ID = os.getenv("AGENT_NODE_ID", "dspy-benchmark-studio")
router = AgentRouter(prefix="", tags=["meta"])


@router.skill(tags=["ops"])
def llm_status(model: Optional[str] = None) -> dict:
    """Return the resolved LLM provider, model, and a best-effort model list."""
    cfg = llm.resolve_config(model)
    try:
        available = llm.list_available_models(cfg)[:50]
    except Exception:
        available = []
    return {
        "provider": cfg.provider,
        "vendor": cfg.vendor,
        "model": cfg.api_model,
        "display_model": cfg.display_model,
        "base_url": cfg.openai_base if cfg.provider == "openai" else cfg.host,
        "available_models": available,
    }


# ---------------------------------------------------------------------------
# meta-level: gap_analyzer
# ---------------------------------------------------------------------------

@router.reasoner()
async def gap_analyzer(
    strategies: list[dict],
    model: Optional[str] = None,
) -> GapAnalysis:
    """Read the 3 strategy scores, decide if retry is needed and how.

    Trigger: any strategy < 0.3 accuracy OR naive beats both DSPy strategies
    by more than 5%.
    """
    by_name = {s["strategy_name"]: s for s in strategies}
    naive_acc = by_name.get("naive", {}).get("accuracy", 0.0)
    predict_acc = by_name.get("dspy_predict", {}).get("accuracy", 0.0)
    cot_acc = by_name.get("dspy_cot", {}).get("accuracy", 0.0)

    weakest = min(
        [("naive", naive_acc), ("dspy_predict", predict_acc), ("dspy_cot", cot_acc)],
        key=lambda x: x[1],
    )
    weakest_name, weakest_acc = weakest

    needs_retry = (
        weakest_acc < 0.3
        or naive_acc > max(predict_acc, cot_acc) + 0.05
    )

    if weakest_name == "dspy_cot":
        recommended = "MIPROv2"
        demos = 6
        trials = 30
    elif weakest_name == "dspy_predict":
        recommended = "BootstrapFewShot"
        demos = 8
        trials = 20
    else:
        recommended = "MIPROv2"
        demos = 4
        trials = 25

    return GapAnalysis(
        weakest_strategy=weakest_name,
        weakest_accuracy=round(weakest_acc, 3),
        needs_retry=needs_retry,
        recommended_optimizer=recommended,
        recommended_max_bootstrapped_demos=demos,
        recommended_num_trials=trials,
        reasoning=(
            f"Weakest strategy: {weakest_name} @ {weakest_acc:.2f}. "
            f"naive={naive_acc:.2f}, dspy_predict={predict_acc:.2f}, dspy_cot={cot_acc:.2f}. "
            f"Retry={'YES' if needs_retry else 'no'}."
        ),
        confident=True,
    )


# ---------------------------------------------------------------------------
# entry: benchmark_orchestrator
# ---------------------------------------------------------------------------

@router.reasoner(tags=["entry"])
async def benchmark_orchestrator(
    task_description: str,
    examples: list[dict],
    model: Optional[str] = None,
    train_ratio: float = 0.6,
) -> BenchmarkReport:
    """Entry reasoner. Coordinates the full benchmark.

    Flow:
      1. Split dataset train/eval.
      2. Fan out 3 strategies in parallel (asyncio.gather).
      3. Call gap_analyzer to read scores.
      4. If gap_analyzer says retry, call enhanced_cot_runner.
      5. Synthesize the final report with winner + improvement %.
    """
    train, eval_set = helpers.train_test_split(examples, train_ratio)

    # Layer 2: 3 strategies in parallel
    naive_t, predict_t, cot_t = await asyncio.gather(
        router.call(
            f"{NODE_ID}.naive_prompting",
            train_examples=train, eval_examples=eval_set, model=model,
            task_description=task_description,
        ),
        router.call(
            f"{NODE_ID}.dspy_predict_optimized",
            train_examples=train, eval_examples=eval_set, model=model,
            task_description=task_description,
        ),
        router.call(
            f"{NODE_ID}.dspy_cot_optimized",
            train_examples=train, eval_examples=eval_set, model=model,
            task_description=task_description,
        ),
    )

    naive = StrategyResult(**naive_t)
    predict = StrategyResult(**predict_t)
    cot = StrategyResult(**cot_t)
    strategies = [naive, predict, cot]

    # Meta-level: gap analysis
    gap_dict = await router.call(
        f"{NODE_ID}.gap_analyzer",
        strategies=[s.model_dump() for s in strategies],
        model=model,
    )
    gap = GapAnalysis(**gap_dict)

    # Dynamic routing: enhanced round if needed
    enhanced: EnhancedStrategyResult | None = None
    if gap.needs_retry:
        enh_dict = await router.call(
            f"{NODE_ID}.enhanced_cot_runner",
            eval_examples=eval_set,
            train_examples=train,
            recommended_optimizer=gap.recommended_optimizer,
            max_bootstrapped_demos=gap.recommended_max_bootstrapped_demos,
            num_trials=gap.recommended_num_trials,
            model=model,
            task_description=task_description,
        )
        # enhanced_cot_runner returns a StrategyResult; wrap as Enhanced
        enh_strat = StrategyResult(**enh_dict)
        enhanced = EnhancedStrategyResult(
            strategy_name=enh_strat.strategy_name + "_enhanced",
            accuracy=enh_strat.accuracy,
            avg_latency_ms=enh_strat.avg_latency_ms,
            total_tokens=enh_strat.total_tokens,
            per_example=enh_strat.per_example,
            confident=enh_strat.confident,
        )
        # Replace the weakest strategy in the report
        for i, s in enumerate(strategies):
            if s.strategy_name == gap.weakest_strategy:
                strategies[i] = enh_strat
                break

    # Winner + improvement
    best = max(strategies, key=lambda s: s.accuracy)
    naive_acc = next(
        (s.accuracy for s in strategies if s.strategy_name == "naive"),
        naive.accuracy,
    )
    improvement_pct = (
        ((best.accuracy - naive_acc) / max(naive_acc, 1e-9)) * 100.0
    )

    cfg = llm.resolve_config(model)
    return BenchmarkReport(
        task_description=task_description,
        model=cfg.display_model,
        provider=cfg.provider,
        strategies=strategies,
        enhanced_round=enhanced,
        gap_analysis=gap,
        winner=best.strategy_name,
        improvement_pct=round(improvement_pct, 1),
        confident=gap.confident and all(s.confident for s in strategies),
    )
