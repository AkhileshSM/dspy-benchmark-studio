"""Reasoner package for dspy-benchmark-studio.

Two routers:
  - strategies: the 3 Q&A strategy reasoners (naive, dspy_predict, dspy_cot) + single_example_runner
  - meta:      orchestrator, optimizer_runner, gap_analyzer, enhanced_cot_runner, confidence_scorer
"""

from .strategies import router as strategies_router
from .meta import router as meta_router

__all__ = ["strategies_router", "meta_router"]
