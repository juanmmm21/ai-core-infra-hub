"""End-to-end RAG evaluation: an ``aicore.rag`` pipeline measured with llm-eval-harness metrics.

The benchmark builds one **open corpus** from every passage of every question (for HotpotQA,
10 paragraphs per question, deduplicated by title), so each question competes with the others'
paragraphs — harder than the per-question *distractor* setting. For each question it retrieves
``k`` passages with the pipeline under test, answers (with the pipeline's generator, or
extractively), and scores exact match, F1, containment, faithfulness and context recall /
precision with the harness.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .._deps import component
from ..types import Document

AREA = "ops"


@dataclass(frozen=True)
class BenchmarkResult:
    """Mean metrics, per-question rows and timings of one run."""

    name: str
    metrics: dict[str, float]
    per_example: list[dict[str, Any]]
    n: int
    corpus_size: int
    ingest_s: float
    total_s: float


def load_hotpotqa(limit: int = 100, offset: int = 0, cache_dir: str | None = None) -> list[Any]:
    """HotpotQA (distractor, validation) as llm-eval-harness ``RAGExample`` objects."""
    return list(
        component(AREA, "llm_eval_harness.datasets").load_hotpotqa(limit=limit, offset=offset, cache_dir=cache_dir)
    )


def load_examples(path: str) -> list[Any]:
    """A JSONL file of ``RAGExample`` rows (as written by the harness)."""
    return list(component(AREA, "llm_eval_harness.datasets").load_jsonl(path))


def corpus_from(examples: Sequence[Any]) -> list[Document]:
    """Every passage of every example, one :class:`Document` per distinct title."""
    seen: dict[str, Document] = {}
    for ex in examples:
        for p in ex.passages:
            if p.title not in seen:
                seen[p.title] = Document(p.title, f"{p.title}. {p.text}", {"title": p.title})
    return list(seen.values())


class RAGBenchmark:
    """Runs a pipeline built by ``factory()`` over ``examples`` and scores it.

    ``factory`` returns a fresh :class:`aicore.rag.RAGPipeline` (so each configuration indexes
    the corpus from scratch and ingestion time is measured too).
    """

    def __init__(self, examples: Sequence[Any], k: int = 5) -> None:
        self.examples = list(examples)
        self.k = k
        self.corpus = corpus_from(self.examples)
        self._metrics = component(AREA, "llm_eval_harness.rag_metrics")

    def run(self, name: str, factory: Callable[[], Any], max_tokens: int = 32) -> BenchmarkResult:
        start = time.perf_counter()
        rag = factory()
        rag.ingest(self.corpus, chunk=False)
        ingest_s = time.perf_counter() - start
        preds = []
        for ex in self.examples:
            t0 = time.perf_counter()
            answer = rag.answer(ex.question, k=self.k, max_tokens=max_tokens)
            hits = list(answer.hits)
            preds.append(
                self._metrics.RAGPrediction(
                    example_id=ex.id,
                    answer=answer.text,
                    retrieved_titles=[str(h.metadata.get("title", h.id)) for h in hits],
                    contexts=[h.text for h in hits],
                    latency_s=time.perf_counter() - t0,
                )
            )
        report = self._metrics.evaluate_rag(self.examples, preds)
        rows = [
            {"id": ex.id, "answer": p.answer, **row}
            for ex, p, row in zip(self.examples, preds, report.per_example, strict=True)
        ]
        return BenchmarkResult(
            name=name,
            metrics=dict(report.metrics),
            per_example=rows,
            n=report.n,
            corpus_size=len(self.corpus),
            ingest_s=ingest_s,
            total_s=time.perf_counter() - start,
        )

    def regressions(self, current: dict[str, float], baseline: dict[str, float], tolerance: float = 0.02) -> list[str]:
        return list(self._metrics.regressions(current, baseline, tolerance))
