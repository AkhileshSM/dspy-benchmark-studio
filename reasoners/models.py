"""Pydantic schemas for dspy-benchmark-studio.

Every schema used by any reasoner lives here so type hints stay consistent
across the cross-call boundary.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------- Input shapes (passed in from the curl / orchestrator) ----------

class QAExample(BaseModel):
    """One reading-comprehension example."""
    passage: str
    question: str
    expected_answer: str = Field(alias="expected_answer")


class BenchmarkRequest(BaseModel):
    """Top-level input to the orchestrator."""
    task_description: str
    examples: list[QAExample]
    model: Optional[str] = None
    # 70/30 train/test split for the optimizer
    train_ratio: float = 0.6


# ---------- Per-strategy output shapes ----------

class ConfidenceScore(BaseModel):
    """Output of the confidence_scorer leaf."""
    correct: bool
    confidence: float
    confident: bool
    method: str  # "substring" | "llm_judge" | "empty" | "fallback_no_match"


class ExampleResult(BaseModel):
    """Result of running one example through one strategy."""
    example_index: int
    predicted_answer: str
    expected_answer: str
    correct: bool
    confidence: float
    latency_ms: float
    prompt_used: str
    token_count: int
    confident: bool
    reasoning: str = ""  # for CoT strategies


class StrategyResult(BaseModel):
    """Aggregate result of one strategy across the full eval set."""
    strategy_name: str
    accuracy: float
    avg_latency_ms: float
    total_tokens: int
    per_example: list[ExampleResult]
    compiled_prompt: str  # what DSPy actually compiled (raw vs optimized diff)
    raw_prompt: str       # the hand-crafted prompt before optimization
    optimizer: str         # "none" | "BootstrapFewShot" | "MIPROv2"
    confident: bool


# ---------- Orchestrator output ----------

class GapAnalysis(BaseModel):
    """Output of gap_analyzer: which strategies underperform, what to retry."""
    weakest_strategy: str
    weakest_accuracy: float
    needs_retry: bool
    recommended_optimizer: str
    recommended_max_bootstrapped_demos: int
    recommended_num_trials: int
    reasoning: str
    confident: bool


class EnhancedStrategyResult(BaseModel):
    """The 'enhanced' CoT + tool-use round if gap_analyzer fires."""
    strategy_name: str
    accuracy: float
    avg_latency_ms: float
    total_tokens: int
    per_example: list[ExampleResult]
    confident: bool


class BenchmarkReport(BaseModel):
    """The final user-facing benchmark report."""
    task_description: str
    model: str
    provider: str = "ollama"  # "ollama" | "openai"
    strategies: list[StrategyResult]
    enhanced_round: Optional[EnhancedStrategyResult] = None
    gap_analysis: Optional[GapAnalysis] = None
    winner: str
    improvement_pct: float  # DSPy best vs naive
    confident: bool
