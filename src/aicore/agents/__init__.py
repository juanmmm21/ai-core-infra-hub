"""Agents: long-term memory, routing, guardrails, sandboxed tools and multi-agent orchestration.

Install with ``pip install "aicore[agents]"``. As everywhere in the SDK, each class wraps one
component and returns :mod:`aicore.types`.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .._deps import component
from ..adapters import agents as adapt
from ..embeddings import Embedder
from ..types import GuardVerdict, MemoryHit, RouteDecision, ToolResult

__all__ = ["Guard", "IntentRouter", "Memory", "Orchestrator", "Router", "ToolRuntime"]

AREA = "agents"


class Memory:
    """agentic-memory-layer: facts with importance and forgetting (``score = similarity *
    importance * exp(-λ·Δt)``), persisted to ``path`` when given.

    ``embedder`` is any :class:`~aicore.embeddings.Embedder` (one encoder for the whole SDK);
    ``vector_store`` optionally puts the facts in a nano-vector-db index instead of the
    component's exact in-memory scan.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        path: str | os.PathLike[str] | None = None,
        half_life_days: float | None = 30.0,
        use_vector_index: bool = False,
        short_term_turns: int = 5,
    ) -> None:
        mod = component(AREA, "agentic_memory_layer")
        store = None
        if use_vector_index:
            if embedder is None:
                raise ValueError("use_vector_index needs an embedder (the index needs a fixed dimension)")
            nvdb = component("rag", "nano_vector_db")
            store = nvdb.NanoVectorDB(dimension=embedder.dimension, metric="cosine", index_type="hnsw")
        self.embedder = embedder
        embed_fn: Callable[[str], Sequence[float]] | None = (lambda t: embedder.embed([t])[0]) if embedder else None
        self._mem = mod.AgenticMemory(
            short_term_max_turns=short_term_turns,
            vector_store=store,
            embed_fn=embed_fn,
            half_life_s=half_life_days * 86400.0 if half_life_days else None,
            decay_factor=0.0,
            path=str(path) if path else None,
        )

    def __len__(self) -> int:
        return len(self._mem)

    def remember(
        self,
        text: str,
        importance: float = 5.0,
        metadata: Mapping[str, Any] | None = None,
        id: str | None = None,
        created_at: float | None = None,
    ) -> str:
        """Store a fact; returns its id (same ``id`` replaces the previous fact)."""
        return str(self._mem.save_fact(text, importance, dict(metadata or {}), id, created_at))

    def remember_many(
        self,
        items: Sequence[tuple[str, str, Mapping[str, Any]]],
        importance: float | Sequence[float] = 5.0,
        created_at: Sequence[float | None] | None = None,
        batch_size: int = 64,
    ) -> int:
        """Bulk ``(id, text, metadata)`` insert: embedded in batches, saved once. ``importance``
        and ``created_at`` may be given per item (same order as ``items``)."""
        n = len(items)
        imps = [float(importance)] * n if isinstance(importance, (int, float)) else [float(i) for i in importance]
        dates = list(created_at) if created_at is not None else [None] * n
        if len(imps) != n or len(dates) != n:
            raise ValueError("importance and created_at must have one value per item")
        vectors: list[Any] = []
        for start in range(0, n, batch_size):
            chunk = [t for _, t, _ in items[start : start + batch_size]]
            vectors.extend(self.embedder.embed(chunk) if self.embedder else [None] * len(chunk))
        with self._mem.batch():
            for (fid, text, meta), vec, imp, when in zip(items, vectors, imps, dates, strict=True):
                self._mem.save_fact(text, imp, dict(meta), fid, when, vec)
        return n

    def recall(
        self,
        query: str,
        k: int = 3,
        where: Mapping[str, Any] | None = None,
        min_similarity: float = 0.0,
        now: float | None = None,
    ) -> list[MemoryHit]:
        rows = self._mem.recall(
            query, top_k=k, current_time=now, min_similarity=min_similarity, where=dict(where) if where else None
        )
        return adapt.memory_hits(rows)

    def forget(self, id: str) -> bool:
        return bool(self._mem.forget(id))

    def ids(self) -> list[str]:
        return [str(f["id"]) for f in self._mem.facts()]

    def add_turn(self, role: str, content: str) -> None:
        """Short-term buffer; older turns are folded into :attr:`summary`."""
        self._mem.add_interaction(role, content)

    @property
    def summary(self) -> str:
        return str(self._mem.conversation_summary)

    def save(self, path: str | os.PathLike[str] | None = None) -> None:
        self._mem.save(str(path) if path else None)


class Router:
    """semantic-model-router: which model (or class) should take a prompt."""

    def __init__(self, profiles: Sequence[Mapping[str, Any]] | None = None, router: Any = None) -> None:
        mod = component(AREA, "semantic_model_router")
        if router is not None:
            self._router = router
        else:
            custom = [mod.ModelProfile(**p) for p in profiles] if profiles else None
            self._router = mod.SemanticModelRouter(custom_models=custom)

    def route(self, prompt: str) -> RouteDecision:
        return adapt.route_from_router(self._router.route(prompt))


class IntentRouter:
    """semantic-model-router's learned intent classifier, served from a champion/challenger
    registry (``registry_dir``) or from a single saved model (``model_dir``).

    ``route(text, context)`` returns the champion's label, its probability as ``confidence`` and
    ``path="champion:<version>"``. With ``shadow=<version>`` a challenger also runs on every
    message and :attr:`shadow_agreement` tells how often it agrees.
    """

    def __init__(
        self,
        registry_dir: str | os.PathLike[str] | None = None,
        model_dir: str | os.PathLike[str] | None = None,
        shadow: str | None = None,
    ) -> None:
        reg = component(AREA, "semantic_model_router.registry")
        from ..adapters.ops import register_router_kinds

        register_router_kinds()
        if (registry_dir is None) == (model_dir is None):
            raise ValueError("give exactly one of registry_dir or model_dir")
        self._cc = None
        if registry_dir is not None:
            self._cc = reg.ChampionChallengerRouter(reg.ModelRegistry(str(registry_dir)), challenger=shadow)
            self.version = str(self._cc.champion_version)
            self._model = self._cc.model
        else:
            self._model = reg.load_model(str(model_dir))
            self.version = os.path.basename(str(model_dir).rstrip("/"))
        self.labels: list[str] = list(self._model.labels)

    def route(self, text: str, context: Sequence[str] | None = None) -> RouteDecision:
        start = time.perf_counter()
        if self._cc is not None:
            out = self._cc.route(text, list(context or []))
            label, confidence = str(out["label"]), float(out["confidence"])
        else:
            p = self._model.predict_proba([text], [list(context or [])])[0]
            best = max(range(len(self.labels)), key=lambda i: p[i])
            label, confidence = self.labels[best], float(p[best])
        ms = (time.perf_counter() - start) * 1000
        return RouteDecision(label=label, reason=f"{ms:.2f} ms", confidence=confidence, path=f"champion:{self.version}")

    @property
    def shadow_agreement(self) -> float | None:
        return None if self._cc is None else self._cc.shadow_agreement


class Guard:
    """llm-guardrails-shield: prompt-injection and toxicity on the way in (PII redacted),
    secret leaks and unsupported claims on the way out."""

    def __init__(self, toxic_keywords: Sequence[str] | None = None, shield: Any = None) -> None:
        mod = component(AREA, "llm_guardrails_shield")
        self._shield = (
            shield
            if shield is not None
            else mod.LLMGuardrailsShield(toxic_keywords=list(toxic_keywords) if toxic_keywords else None)
        )

    def check_input(self, text: str) -> GuardVerdict:
        return adapt.verdict_from_shield(self._shield.validate_input(text))

    def check_output(self, text: str, context: str | None = None) -> GuardVerdict:
        return adapt.verdict_from_shield(self._shield.validate_output(text, context))

    def redact(self, text: str) -> str:
        return str(self._shield.redact_pii(text))


class ToolRuntime:
    """secure-tool-runtime: Python checked by AST rules, then run in a separate process."""

    def __init__(self, banned_modules: set[str] | None = None, banned_calls: set[str] | None = None) -> None:
        mod = component(AREA, "secure_tool_runtime")
        self._rt = mod.SecureToolRuntime(banned_modules=banned_modules, banned_calls=banned_calls)

    def run(self, code: str, timeout_s: float = 2.0) -> ToolResult:
        start = time.perf_counter()
        result = self._rt.run_code(code, timeout_seconds=timeout_s)
        return adapt.tool_result(result, time.perf_counter() - start)


class Orchestrator:
    """orchestra-agents: a supervisor that splits a task among registered ReAct agents.
    Tools are plain ``str -> str`` callables. The component's planner delegates to agents named
    ``ResearcherAgent`` and ``WriterAgent``; register agents under those names."""

    def __init__(self, name: str = "CentralSupervisor") -> None:
        self._mod = component(AREA, "orchestra_agents")
        self._orch = self._mod.AgentOrchestrator(name=name)

    def add_agent(
        self, name: str, role: str, system_prompt: str, tools: Mapping[str, Callable[[str], str]] | None = None
    ) -> None:
        agent = self._mod.AutonomousAgent(name=name, role=role, system_prompt=system_prompt, tools=dict(tools or {}))
        self._orch.register_agent(agent)

    def plan(self, task: str) -> list[dict[str, Any]]:
        return [d.model_dump() for d in self._orch.plan_delegation(task)]

    def run(self, task: str) -> str:
        return str(self._orch.execute(task))
