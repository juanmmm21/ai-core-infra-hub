"""AI Core SDK: one stable, typed API over the ai-core components.

Areas are separate extras and separate subpackages, imported only when used::

    pip install "aicore[rag]"      # aicore.rag     tokenizer, chunker, parser, vectors, hybrid search, reranker, graph
    pip install "aicore[agents]"   # aicore.agents  memory, router, guardrails, sandbox, orchestrator
    pip install "aicore[ops]"      # aicore.ops     inference client, tracer, evaluator, synthetic data, datasets
    pip install "aicore[train]"    # aicore.train   LoRA/QLoRA fine-tuning (PyTorch)

The top level holds what every area shares: the result types, the embedders and the errors.
What is exported here and from each area's ``__all__`` is the stable API (``docs/sdk.md``):
it only changes in a new minor version, additively, until 1.0.
"""

from __future__ import annotations

from .embeddings import Embedder, HashEmbedder, OllamaEmbedder, SentenceTransformerEmbedder, cosine
from .errors import AicoreError, InferenceError, MissingExtraError
from .types import (
    Answer,
    Chunk,
    Document,
    Generation,
    GuardVerdict,
    Hit,
    MemoryHit,
    RouteDecision,
    ToolResult,
    Triple,
)

__version__ = "0.1.0"

__all__ = [
    "AicoreError",
    "Answer",
    "Chunk",
    "Document",
    "Embedder",
    "Generation",
    "GuardVerdict",
    "HashEmbedder",
    "Hit",
    "InferenceError",
    "MemoryHit",
    "MissingExtraError",
    "OllamaEmbedder",
    "RouteDecision",
    "SentenceTransformerEmbedder",
    "ToolResult",
    "Triple",
    "__version__",
    "cosine",
]
