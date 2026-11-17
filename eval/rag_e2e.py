"""End-to-end RAG evaluation of the AI Core pipeline on HotpotQA (open corpus).

    python eval/rag_e2e.py --data eval/data/hotpotqa_distractor_dev_50.jsonl --configs bm25,hybrid-hash
    python eval/rag_e2e.py --n 200 --configs bm25,hybrid-hash,hybrid-minilm,hybrid-minilm-rerank
    python eval/rag_e2e.py --n 200 --configs hybrid-minilm-rerank-llm \
        --llm-url http://127.0.0.1:8000/v1 --llm-model qwen2.5-0.5b
    python eval/rag_e2e.py --data eval/data/hotpotqa_distractor_dev_50.jsonl --configs hybrid-hash \
        --gate eval/baseline_ci.json                 # the CI job: exit 1 on regression

Configurations (all through the aicore SDK):

* ``bm25``: sparse retrieval only, extractive answer.
* ``hybrid-hash``: BM25 + dense (``HashEmbedder``) fused by RRF, extractive answer. Offline and
  deterministic: this is the one the CI gates.
* ``hybrid-minilm``: dense side with ``sentence-transformers/all-MiniLM-L6-v2``.
* ``hybrid-minilm-rerank``: + ``cross-encoder/ms-marco-MiniLM-L-6-v2`` over the top 15.
* ``hybrid-minilm-rerank-llm``: + an LLM behind any OpenAI-compatible server writes the answer.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from aicore import HashEmbedder, SentenceTransformerEmbedder
from aicore.ops import InferenceClient, RAGBenchmark, load_examples, load_hotpotqa
from aicore.rag import HybridRetriever, RAGPipeline, Reranker

HERE = Path(__file__).resolve().parent
KEYS = ("exact_match", "f1", "contains", "faithfulness", "context_recall", "context_precision", "latency_s")

LLM_PROMPT = (
    "Answer the question using only the context. Reply with the answer only: a few words, "
    "a name, a date or yes/no. No explanation.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
)


class SparseOnly(HybridRetriever):
    """BM25 alone, through the same interface."""

    def search(self, query: str, k: int = 5, fusion: Any = "rrf", alpha: float = 0.5) -> list[Any]:
        return self.sparse_search(query, k)


def factories(args: argparse.Namespace) -> dict[str, Callable[[], RAGPipeline]]:
    cache: dict[str, Any] = {}

    def minilm() -> SentenceTransformerEmbedder:
        if "minilm" not in cache:
            cache["minilm"] = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
        return cache["minilm"]

    def reranker() -> Reranker:
        if "rerank" not in cache:
            cache["rerank"] = Reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
            if not cache["rerank"].uses_model:
                sys.exit("the cross-encoder did not load (install cross-encoder-reranker[model])")
        return cache["rerank"]

    def llm() -> InferenceClient:
        if not args.llm_url:
            sys.exit("--llm-url is required for the -llm configuration")
        return InferenceClient(args.llm_url, model=args.llm_model)

    hash_emb = HashEmbedder(512)
    return {
        "bm25": lambda: RAGPipeline(hash_emb, retriever=SparseOnly(hash_emb)),
        "hybrid-hash": lambda: RAGPipeline(hash_emb),
        "hybrid-minilm": lambda: RAGPipeline(minilm()),
        "hybrid-minilm-rerank": lambda: RAGPipeline(minilm(), reranker=reranker()),
        "hybrid-minilm-rerank-llm": lambda: RAGPipeline(
            minilm(), reranker=reranker(), generator=llm(), prompt=LLM_PROMPT
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="JSONL of RAGExample rows (default: download --n from HotpotQA)")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--configs", default="bm25,hybrid-hash")
    parser.add_argument("--llm-url")
    parser.add_argument("--llm-model", default="default")
    parser.add_argument("--gate", help="baseline JSON: exit 1 if a metric of the first config regresses")
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--write-baseline", help="write the first config's metrics as a baseline")
    parser.add_argument("--out", help="write all results as JSON")
    args = parser.parse_args(argv)

    examples = (
        load_examples(args.data)
        if args.data
        else load_hotpotqa(limit=args.n, cache_dir=os.environ.get("AICORE_EVAL_CACHE"))
    )
    bench = RAGBenchmark(examples, k=args.k)
    table = factories(args)
    print(
        f"HotpotQA dev (distractor paragraphs pooled into one open corpus): {len(examples)} questions, "
        f"{len(bench.corpus)} passages, top-{args.k} · Python {platform.python_version()} · {platform.machine()} · "
        f"{os.cpu_count()} CPUs"
    )
    print("| Configuration | EM | F1 | contains | faithfulness | context recall | context precision | s/question |")
    print("|---|---|---|---|---|---|---|---|")
    results = []
    for name in args.configs.split(","):
        result = bench.run(name, table[name])
        results.append(result)
        m = result.metrics
        print(f"| {name} | " + " | ".join(f"{m.get(k, 0.0):.3f}" for k in KEYS) + " |", flush=True)

    first = {k: round(v, 4) for k, v in results[0].metrics.items() if k != "latency_s"}
    if args.out:
        payload = {
            "machine": {"python": platform.python_version(), "arch": platform.machine(), "cpus": os.cpu_count()},
            "n": len(examples),
            "k": args.k,
            "corpus": len(bench.corpus),
            "when": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": [
                {
                    "name": r.name,
                    "metrics": r.metrics,
                    "ingest_s": r.ingest_s,
                    "total_s": r.total_s,
                    "per_example": r.per_example,
                }
                for r in results
            ],
        }
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.write_baseline:
        Path(args.write_baseline).write_text(json.dumps(first, indent=2), encoding="utf-8")
    if args.gate:
        problems = bench.regressions(first, json.loads(Path(args.gate).read_text()), args.tolerance)
        if problems:
            print("REGRESSION:\n  " + "\n  ".join(problems))
            return 1
        print(f"No regressions against {args.gate}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
