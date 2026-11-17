"""Retrieval-augmented generation: parse, chunk, embed, index, retrieve, rerank, answer.

Install with ``pip install "aicore[rag]"``. Every class wraps one component and speaks only
:mod:`aicore.types`; :class:`RAGPipeline` wires them together. Without arguments everything runs
offline (BPE-free word counting, :class:`~aicore.embeddings.HashEmbedder`, lexical reranker);
pass real models to get real quality.
"""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

from .._deps import component
from ..adapters import rag as adapt
from ..embeddings import Embedder, HashEmbedder
from ..types import Answer, Chunk, Document, Hit, Triple

__all__ = [
    "BPETokenizer",
    "Chunker",
    "DocumentParser",
    "Generator",
    "GraphExtractor",
    "HybridRetriever",
    "RAGPipeline",
    "Reranker",
    "VectorIndex",
]

AREA = "rag"


class BPETokenizer:
    """bpe-tokenizer-from-scratch: train a byte-level BPE vocabulary, encode, decode, count."""

    def __init__(self) -> None:
        mod = component(AREA, "bpe_tokenizer_from_scratch")
        self._tok = mod.BPETokenizer()

    def train(self, text: str, vocab_size: int = 300) -> BPETokenizer:
        self._tok.train(text, vocab_size)
        return self

    def encode(self, text: str) -> list[int]:
        return list(self._tok.encode(text))

    def decode(self, ids: Sequence[int]) -> str:
        return str(self._tok.decode(list(ids)))

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def pieces(self, text: str) -> list[str]:
        """The text of each token, for display."""
        return [self.decode([i]) for i in self.encode(text)]

    def save(self, path: str | os.PathLike[str]) -> None:
        self._tok.save(str(path))

    def load(self, path: str | os.PathLike[str]) -> BPETokenizer:
        self._tok.load(str(path))
        return self


class _Encoder(Protocol):
    def encode(self, text: str) -> list[int]: ...


class Chunker:
    """semantic-chunking-engine: splits where the meaning shifts, within ``max_tokens``."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        tokenizer: _Encoder | None = None,
        window_size: int = 3,
        threshold: float = 1.2,
        max_tokens: int = 512,
    ) -> None:
        mod = component(AREA, "semantic_chunking_engine")
        self.embedder = embedder if embedder is not None else HashEmbedder()
        self._chunker = mod.SemanticChunker(
            embedding_provider=adapt.EmbeddingProviderAdapter(self.embedder),
            tokenizer=tokenizer,
            window_size=window_size,
            threshold_factor=threshold,
            max_tokens=max_tokens,
        )

    def split(self, document: Document | str) -> list[Chunk]:
        doc = document if isinstance(document, Document) else Document("doc", document)
        return adapt.chunks_from_chunker(doc, self._chunker.chunk_text(doc.text))


class DocumentParser:
    """multimodal-doc-parser: PDFs and images to text with the local extractor (PyMuPDF)."""

    def __init__(self, backend: Any = None) -> None:
        mod = component(AREA, "multimodal_doc_parser")
        self._parser = mod.MultimodalDocParser(backend or mod.LocalExtractBackend())

    def parse(self, path: str | os.PathLike[str], document_id: str | None = None) -> Document:
        p = Path(path)
        if p.suffix.lower() in {".txt", ".md", ".rst", ".csv", ".json", ".py"}:
            text = p.read_text(encoding="utf-8", errors="replace")
        else:
            text = str(self._parser.parse(str(p)))
        return Document(document_id or p.name, text, {"source": str(p)})


class VectorIndex:
    """nano-vector-db: exact or HNSW nearest-neighbour search over texts or vectors."""

    def __init__(
        self,
        dimension: int,
        metric: Literal["cosine", "l2", "dot"] = "cosine",
        index_type: Literal["hnsw", "flat"] = "hnsw",
        db: Any = None,
    ) -> None:
        mod = component(AREA, "nano_vector_db")
        self.metric = metric
        self.db = db if db is not None else mod.NanoVectorDB(dimension=dimension, metric=metric, index_type=index_type)
        self.texts: dict[str, str] = {}

    @property
    def dimension(self) -> int:
        return int(self.db.dimension)

    def __len__(self) -> int:
        return len(self.db.vectors)

    def add(self, id: str, vector: Sequence[float], text: str = "", metadata: dict[str, Any] | None = None) -> None:
        self.db.insert(id=id, vector=list(vector), metadata=dict(metadata or {}))
        self.texts[id] = text

    def search(self, vector: Sequence[float], k: int = 5, where: dict[str, Any] | None = None) -> list[Hit]:
        rows = self.db.query(list(vector), top_k=k, filter=where)
        return adapt.hits_from_vector_results(rows, self.texts, self.metric)

    def save(self, path: str | os.PathLike[str]) -> None:
        self.db.save(str(path))

    @classmethod
    def load(cls, path: str | os.PathLike[str], texts: dict[str, str] | None = None) -> VectorIndex:
        mod = component(AREA, "nano_vector_db")
        db = mod.NanoVectorDB.load(str(path))
        index = cls(int(db.dimension), metric=db.metric, db=db)
        index.texts = dict(texts or {})
        return index


class HybridRetriever:
    """hybrid-search-retrieval-pipeline over a :class:`VectorIndex`: BM25 + dense, fused by RRF
    (or by normalized scores with ``fusion="score"``)."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        index: VectorIndex | None = None,
        index_type: Literal["hnsw", "flat"] = "flat",
    ) -> None:
        mod = component(AREA, "hybrid_search_retrieval_pipeline")
        self.embedder = embedder if embedder is not None else HashEmbedder()
        self.index = index if index is not None else VectorIndex(self.embedder.dimension, index_type=index_type)
        self._mod = mod
        self._corpus: dict[str, str] = dict(self.index.texts)
        self._pipeline = mod.HybridSearchPipeline(
            vector_db=self.index.db, embed_fn=lambda q: self.embedder.embed([q])[0]
        )
        self._dirty = bool(self._corpus)
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._corpus)

    def add(self, chunks: Iterable[Chunk], batch_size: int = 64) -> int:
        """Embed and index ``chunks``; BM25 is refitted lazily before the next search."""
        items = list(chunks)
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            vectors = self.embedder.embed([c.text for c in batch])
            for chunk, vector in zip(batch, vectors, strict=True):
                meta = {**chunk.metadata, "document_id": chunk.document_id, "chunk_index": chunk.index}
                self.index.add(chunk.id, vector, chunk.text, meta)
                self._corpus[chunk.id] = chunk.text
        self._dirty = True
        return len(items)

    def _fit(self) -> None:
        with self._lock:
            if self._dirty:
                self._pipeline.fit_sparse(self._corpus or {"_empty": ""})
                self._dirty = False

    def search(self, query: str, k: int = 5, fusion: Literal["rrf", "score"] = "rrf", alpha: float = 0.5) -> list[Hit]:
        if not self._corpus:
            return []
        self._fit()
        rows = self._pipeline.search(query=query, top_k=k, fusion_method=fusion, alpha=alpha)
        return adapt.hits_from_hybrid(rows, self._corpus)

    def sparse_search(self, query: str, k: int = 5) -> list[Hit]:
        """BM25 alone (useful as a baseline in evaluations)."""
        if not self._corpus:
            return []
        self._fit()
        rows = self._pipeline.bm25.retrieve(query, top_k=k)
        meta = self.index.db.metadata
        return [
            Hit(str(i), self._corpus.get(str(i), ""), float(s), dict(meta.get(i, {})), {"bm25": float(s)}) for s, i in rows
        ]

    def dense_search(self, query: str, k: int = 5) -> list[Hit]:
        """Vectors alone."""
        return self.index.search(self.embedder.embed([query])[0], k)


class Reranker:
    """cross-encoder-reranker: rescoring of (query, passage) pairs. Without the ``[model]``
    extra of the component (torch) it uses its offline lexical scorer."""

    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str | None = None, offline: bool = False
    ) -> None:
        mod = component(AREA, "cross_encoder_reranker")
        if offline:
            saved = mod.reranker.DL_AVAILABLE
            mod.reranker.DL_AVAILABLE = False
            try:
                self._rr = mod.CrossEncoderReranker(model_name=model_name, device=device)
            finally:
                mod.reranker.DL_AVAILABLE = saved
        else:
            self._rr = mod.CrossEncoderReranker(model_name=model_name, device=device)

    @property
    def uses_model(self) -> bool:
        return bool(getattr(self._rr, "is_online", False))

    def rerank(self, query: str, hits: Sequence[Hit], k: int = 3) -> list[Hit]:
        if not hits:
            return []
        rows = self._rr.rerank(query, adapt.rerank_inputs(hits), top_k=k, text_key="text")
        return adapt.hits_from_rerank(rows)


class GraphExtractor:
    """knowledge-graph-extractor: entities and relations as :class:`Triple`. With a Gemini
    key it asks the model; without one it uses the component's offline extractor."""

    def __init__(self, api_key: str | None = None, model_name: str = "gemini-1.5-flash") -> None:
        mod = component(AREA, "knowledge_graph_extractor")
        self._ex = mod.KnowledgeGraphExtractor(api_key=api_key, model_name=model_name)

    def extract(self, text: str) -> list[Triple]:
        return adapt.triples_from_graph(self._ex.extract(text))

    def extract_graph(self, text: str) -> dict[str, Any]:
        """The raw graph as plain dicts (``entities``, ``relations``) for visualizers."""
        graph = self._ex.extract(text)
        return {
            "entities": [e.model_dump() for e in graph.entities],
            "relations": [r.model_dump() for r in graph.relations],
        }


class Generator(Protocol):
    """Anything that completes a prompt; :class:`aicore.ops.InferenceClient` is one."""

    def complete(self, prompt: str, *, max_tokens: int = ..., temperature: float = ...) -> str: ...


DEFAULT_PROMPT = (
    "Answer the question using only the context. Answer with a short phrase, no explanation.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class RAGPipeline:
    """Ingest documents, retrieve with :class:`HybridRetriever`, optionally rerank, and answer
    with a :class:`Generator` (or, without one, return the best supporting sentence)."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        generator: Generator | None = None,
        reranker: Reranker | None = None,
        chunker: Chunker | None = None,
        retriever: HybridRetriever | None = None,
        prompt: str = DEFAULT_PROMPT,
        candidates: int = 15,
    ) -> None:
        self.embedder = embedder if embedder is not None else HashEmbedder()
        self.retriever = retriever if retriever is not None else HybridRetriever(self.embedder)
        self.chunker = chunker
        self.reranker = reranker
        self.generator = generator
        self.prompt = prompt
        self.candidates = candidates

    def ingest(self, documents: Iterable[Document], chunk: bool = True) -> int:
        """Split (with the chunker, if any) and index ``documents``. Returns chunks indexed."""
        chunks: list[Chunk] = []
        for doc in documents:
            if chunk and self.chunker is not None:
                chunks.extend(self.chunker.split(doc))
            else:
                meta = {**doc.metadata, "document_id": doc.id}
                chunks.append(Chunk(f"{doc.id}#0", doc.id, doc.text, 0, meta))
        return self.retriever.add(chunks)

    def retrieve(self, question: str, k: int = 5) -> list[Hit]:
        hits = self.retriever.search(question, k=max(k, self.candidates) if self.reranker else k)
        if self.reranker is not None:
            hits = self.reranker.rerank(question, hits, k=k)
        return hits[:k]

    def build_prompt(self, question: str, hits: Sequence[Hit]) -> str:
        context = "\n\n".join(f"[{i + 1}] {h.text}" for i, h in enumerate(hits))
        return self.prompt.format(context=context, question=question)

    def answer(self, question: str, k: int = 5, max_tokens: int = 64) -> Answer:
        hits = self.retrieve(question, k)
        if self.generator is None:
            return Answer(self.extractive_answer(question, hits), tuple(hits), "extractive")
        text = self.generator.complete(self.build_prompt(question, hits), max_tokens=max_tokens, temperature=0.0)
        return Answer(text.strip(), tuple(hits), getattr(self.generator, "model", ""))

    def extractive_answer(self, question: str, hits: Sequence[Hit]) -> str:
        """No generator: the retrieved sentence that best covers the question's content words
        (words of three letters or more), preferring higher-ranked hits."""
        words = {w for w in re.findall(r"\w+", question.lower()) if len(w) > 2}
        best, best_score = "", 0.0
        for rank, hit in enumerate(hits):
            for sentence in _SENTENCE.split(hit.text):
                overlap = len(words & set(re.findall(r"\w+", sentence.lower())))
                score = overlap / (1.0 + 0.25 * rank)
                if score > best_score:
                    best, best_score = sentence.strip(), score
        return best or (hits[0].text if hits else "")
