"""The SDK's own types: the only shapes a caller of `aicore` ever sees.

Each component speaks its own dialect (dicts with ``distance``, pydantic models, tuples). The
adapters in :mod:`aicore.adapters` translate those into these frozen dataclasses, so a change
inside a component never reaches a caller, and no component has to import another one to share
a type. Everything here is part of the stable API (see ``docs/sdk.md``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

Metadata = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Document:
    """A source text to ingest, with an id the caller chooses (a path, a URL, a key)."""

    id: str
    text: str
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A piece of a document, the unit that is embedded, indexed and retrieved."""

    id: str
    document_id: str
    text: str
    index: int
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Hit:
    """A retrieved chunk. ``score`` is higher-is-better whatever the retriever measured;
    ``signals`` keeps the raw per-retriever values (``bm25``, ``dense_distance``, ``rerank``)."""

    id: str
    text: str
    score: float
    metadata: Metadata = field(default_factory=dict)
    signals: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Answer:
    """What a RAG pipeline answers, with the evidence it was given."""

    text: str
    hits: tuple[Hit, ...]
    model: str = ""


@dataclass(frozen=True, slots=True)
class Triple:
    """A knowledge-graph edge: ``subject --relation--> object``."""

    subject: str
    relation: str
    object: str
    subject_type: str = ""
    object_type: str = ""


@dataclass(frozen=True, slots=True)
class MemoryHit:
    """A recalled long-term memory with its final score (similarity, importance, decay)."""

    id: str
    text: str
    score: float
    importance: float
    created_at: float
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Where a prompt should go. ``label`` is the class or model chosen; ``confidence`` is in
    [0, 1] when the router can estimate it; ``path`` says which mechanism decided."""

    label: str
    reason: str = ""
    confidence: float | None = None
    path: str = ""
    estimated_cost_per_1k_tokens: float | None = None


@dataclass(frozen=True, slots=True)
class GuardVerdict:
    """A guardrail decision. ``text`` is the sanitized text when allowed (PII redacted) and
    empty when blocked; ``category`` names the rule that blocked it."""

    allowed: bool
    text: str
    reason: str = ""
    category: str = ""


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The outcome of running code in the sandbox."""

    ok: bool
    stdout: str
    stderr: str
    error: str
    duration_s: float


@dataclass(frozen=True, slots=True)
class Generation:
    """A completion from an inference backend, with its token usage when reported."""

    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str = ""
    latency_s: float = 0.0
    time_to_first_token_s: float | None = None
