"""RAG translations: chunker, vector store, hybrid retriever, reranker, graph extractor."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..embeddings import Embedder
from ..types import Chunk, Document, Hit, Triple
from . import check_fields


class EmbeddingProviderAdapter:
    """An :class:`~aicore.embeddings.Embedder` seen as semantic-chunking-engine's
    ``EmbeddingProvider`` (``get_embeddings``)."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.dimension = embedder.dimension

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.embed(texts)


def chunks_from_chunker(document: Document, raw: Sequence[Mapping[str, Any]]) -> list[Chunk]:
    """semantic-chunking-engine dicts (``text``, ``index``, ``tokens_count``) -> :class:`Chunk`."""
    chunks: list[Chunk] = []
    for position, item in enumerate(raw):
        index = int(item.get("index", position))
        meta = dict(document.metadata)
        meta["document_id"] = document.id
        if item.get("tokens_count") is not None:
            meta["tokens"] = int(item["tokens_count"])
        chunks.append(Chunk(f"{document.id}#{index}", document.id, str(item["text"]), index, meta))
    return chunks


def distance_to_score(distance: float, metric: str) -> float:
    """nano-vector-db returns distances (lower is better); hits carry higher-is-better scores."""
    if metric == "cosine":
        return 1.0 - float(distance)
    if metric == "dot":
        return -float(distance)
    return 1.0 / (1.0 + float(distance))


def hits_from_vector_results(results: Sequence[Mapping[str, Any]], texts: Mapping[str, str], metric: str) -> list[Hit]:
    """nano-vector-db ``query`` rows (``id``, ``distance``, ``metadata``) -> :class:`Hit`."""
    hits = []
    for row in results:
        doc_id = str(row["id"])
        distance = float(row["distance"])
        hits.append(
            Hit(
                doc_id,
                texts.get(doc_id, ""),
                distance_to_score(distance, metric),
                dict(row.get("metadata") or {}),
                {"dense_distance": distance},
            )
        )
    return hits


def hits_from_hybrid(results: Sequence[Mapping[str, Any]], texts: Mapping[str, str]) -> list[Hit]:
    """hybrid-search-retrieval-pipeline rows (``score``, ``sparse_score``, ``dense_distance``)."""
    hits = []
    for row in results:
        doc_id = str(row["id"])
        signals = {"fused": float(row["score"]), "bm25": float(row.get("sparse_score") or 0.0)}
        if row.get("dense_distance") is not None:
            signals["dense_distance"] = float(row["dense_distance"])
        hits.append(Hit(doc_id, texts.get(doc_id, ""), float(row["score"]), dict(row.get("metadata") or {}), signals))
    return hits


def rerank_inputs(hits: Sequence[Hit]) -> list[dict[str, Any]]:
    """:class:`Hit` -> the dicts cross-encoder-reranker sorts in place (``text`` key)."""
    return [{"id": h.id, "text": h.text, "_hit": h} for h in hits]


def hits_from_rerank(rows: Sequence[Mapping[str, Any]]) -> list[Hit]:
    """Reranked dicts -> :class:`Hit`, with the cross-encoder score as the new ``score``."""
    out = []
    for row in rows:
        hit: Hit = row["_hit"]
        score = float(row["rerank_score"])
        out.append(Hit(hit.id, hit.text, score, hit.metadata, {**hit.signals, "rerank": score}))
    return out


def triples_from_graph(graph: Any) -> list[Triple]:
    """knowledge-graph-extractor ``KnowledgeGraph`` (pydantic) -> :class:`Triple`."""
    check_fields(type(graph), ("entities", "relations"), "triples_from_graph")
    types = {e.name: e.type for e in graph.entities}
    return [
        Triple(r.source, r.type, r.target, types.get(r.source, ""), types.get(r.target, "")) for r in graph.relations
    ]
