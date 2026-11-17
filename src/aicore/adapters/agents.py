"""Agent translations: guardrails, router, memory and sandbox results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..types import GuardVerdict, MemoryHit, RouteDecision, ToolResult
from . import check_fields

# The shield answers with a sentence; the SDK exposes a stable category instead.
_GUARD_CATEGORIES = (
    ("injection", "prompt_injection"),
    ("jailbreak", "prompt_injection"),
    ("ofensivo", "toxicity"),
    ("inapropiado", "toxicity"),
    ("fuga", "secret_leak"),
    ("credencial", "secret_leak"),
    ("alucinacion", "hallucination"),
    ("consistencia", "hallucination"),
)


def guard_category(reason: str) -> str:
    lowered = reason.lower()
    for needle, category in _GUARD_CATEGORIES:
        if needle in lowered:
            return category
    return "blocked" if reason else ""


def verdict_from_shield(result: tuple[bool, str, str]) -> GuardVerdict:
    """llm-guardrails-shield ``(is_safe, clean_text, reason)`` -> :class:`GuardVerdict`."""
    allowed, text, reason = result
    return GuardVerdict(bool(allowed), text if allowed else "", reason, "" if allowed else guard_category(reason))


def route_from_router(decision: Any) -> RouteDecision:
    """semantic-model-router ``RoutingDecision`` (pydantic) -> :class:`RouteDecision`."""
    check_fields(type(decision), ("selected_model", "reason", "routing_path"), "route_from_router")
    return RouteDecision(
        label=decision.selected_model,
        reason=decision.reason,
        confidence=getattr(decision, "confidence", None),
        path=decision.routing_path,
        estimated_cost_per_1k_tokens=getattr(decision, "estimated_cost_per_1k_tokens", None),
    )


def memory_hits(rows: Sequence[Mapping[str, Any]]) -> list[MemoryHit]:
    """agentic-memory-layer ``recall`` dicts -> :class:`MemoryHit`."""
    return [
        MemoryHit(
            id=str(row.get("id", "")),
            text=str(row["fact"]),
            score=float(row["recall_score"]),
            importance=float(row["importance"]),
            created_at=float(row["created_at"]),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in rows
    ]


def tool_result(result: tuple[bool, str, str, str], duration_s: float) -> ToolResult:
    """secure-tool-runtime ``(ok, stdout, stderr, error)`` -> :class:`ToolResult`."""
    ok, stdout, stderr, error = result
    return ToolResult(bool(ok), stdout or "", stderr or "", error or "", duration_s)
