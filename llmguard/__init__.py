"""llm-guard — self-hosted LLM cost attribution and budget enforcement gateway.

Answers one question with evidence: *where did our LLM spend go, and what
should we change?* Runs on the Python standard library alone.
"""

from __future__ import annotations

__version__ = "0.4.0"

from .analytics import build_report, evaluate_guard
from .pricing import TokenUsage, compute_cost, get_price
from .storage import RequestRecord, Store

__all__ = [
    "__version__",
    "build_report",
    "evaluate_guard",
    "TokenUsage",
    "compute_cost",
    "get_price",
    "RequestRecord",
    "Store",
]
