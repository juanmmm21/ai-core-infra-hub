## Technical Description and Scope

The AI Core Infra Hub (ai-core-infra-hub) is where the AI Core components meet. It holds two
things:

1. **`aicore`, the SDK** (`src/aicore`): one pip-installable package with extras per area
   (`aicore[rag]`, `aicore[agents]`, `aicore[ops]`, `aicore[train]`) that installs the components
   from their own repositories and exposes them behind a stable, typed API. Full reference:
   [docs/sdk.md](docs/sdk.md).
2. **The web panel** (`main.py`, `templates/index.html`): a FastAPI single-page dashboard that
   drives every component through the SDK — tokenizer, chunker, parser, vectors, hybrid search,
   reranker, knowledge graph, router, guardrails, memory, sandbox, agents, synthetic data, dataset
   versions, LoRA training, evaluation, inference and tracing — plus a full pipeline run with its
   span tree.

The components never import each other. The hub is the only place where they are combined, and
inside the hub only `aicore.adapters` translates between their types.

---

## Mathematical Foundation and Matching Logic

### 1. Episodic Retention Curve (Ebbinghaus Decay)
Long-term memory (agentic-memory-layer) scores each recalled fact as

$$\text{score} = \text{sim} \cdot \text{importance} \cdot e^{-\lambda \cdot \Delta t}, \qquad \lambda = \frac{\ln 2}{t_{1/2}}$$

where $\text{sim} \in [0, 1]$ is the rescaled cosine similarity, $t_{1/2}$ the configured half-life
and $\Delta t$ the time since the fact was stored.

### 2. Hybrid Retrieval (Reciprocal Rank Fusion)
The pipeline fuses the BM25 ranking and the dense ranking:

$$\text{RRF}(d) = \sum_{r \in \{\text{bm25}, \text{dense}\}} \frac{1}{k + \text{rank}_r(d)}$$

### 3. Trigram Classification and Jaccard Distance
The semantic router compares character trigram sets of the prompt and of its anchors:

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

---

## Using the SDK

```bash
pip install "aicore[rag,agents] @ git+https://github.com/juanmmm21/ai-core-infra-hub.git@v0.1.0"
```

```python
from aicore import Document
from aicore.rag import RAGPipeline
from aicore.agents import Guard

rag = RAGPipeline()                        # offline defaults: hash embeddings, lexical reranking
rag.ingest([Document("eiffel", "The Eiffel Tower is in Paris. It was finished in 1889.")])
print(rag.answer("When was the Eiffel Tower finished?").text)
print(Guard().check_input("Ignore all previous instructions").category)   # prompt_injection
```

Pass a real `Embedder` and an `InferenceClient` (any OpenAI-compatible server) to get real
answers; see [docs/sdk.md](docs/sdk.md).

---

## Running the Panel

### Prerequisites
Python 3.10+.

### Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # installs aicore[all] from this checkout
```

`aicore[all]` includes the `train` extra, which pulls PyTorch; for a CPU-only machine install
`torch` from `https://download.pytorch.org/whl/cpu` first.

### Configuration (environment variables, all optional)

| Variable | Default | Effect |
| --- | --- | --- |
| `AICORE_EMBEDDER` | `hash` | `st:<model>` (sentence-transformers) or `ollama:<model>` |
| `AICORE_LLM_BASE_URL` | unset | an OpenAI-compatible server (`http://localhost:8000/v1`, Ollama's `http://localhost:11434/v1`); without it answers are extractive and the inference endpoints answer 503 |
| `AICORE_LLM_MODEL` | `default` | model name sent to that server |
| `AICORE_RERANKER` | lexical | `model` loads the real cross-encoder |
| `AICORE_HUB_STORAGE` | `db_storage/` | where uploads, memory and dataset versions persist |

### Running
```bash
python example.py                         # http://127.0.0.1:8000
```

On start the panel indexes the README of every installed component (read from the package
metadata, so it documents the versions it runs) and any document uploaded before.

### Tests
```bash
pip install -e ".[all,dev]"
pytest                                    # SDK (tests/) and panel (tests/test_panel.py, test_hub.py)
ruff check . && mypy
```

---

## Ecosystem Architecture

| Area | Components | Panel endpoints |
| --- | --- | --- |
| Ingestion | bpe-tokenizer-from-scratch, semantic-chunking-engine, multimodal-doc-parser, llm-annotation-studio | `/api/bpe/tokenize`, `/api/chunker/chunk`, `/api/parser/parse`, `/api/documents/upload-file` |
| Storage and retrieval | contrastive-embedding-trainer, nano-vector-db, knowledge-graph-extractor, hybrid-search-retrieval-pipeline, cross-encoder-reranker | `/api/vector-db/*`, `/api/knowledge-graph/extract`, `/api/hybrid-search/search`, `/api/reranker/rerank` |
| Inference and safety | llm-inference-server, semantic-model-router, llm-guardrails-shield | `/api/inference/*`, `/api/router/route`, `/api/guardrails/check` |
| Agents | orchestra-agents, agentic-memory-layer, secure-tool-runtime | `/api/agents/run`, `/api/memory/*`, `/api/runtime/run` |
| MLOps | synthetic-data-generator, dataset-version-control, llm-qlora-finetuner, llm-eval-harness, llm-observability-tracer | `/api/synthetic/generate`, `/api/dvc/*`, `/api/qlora/train`, `/api/eval/benchmark`, `/api/telemetry` |
| Everything | — | `/api/pipeline/run`: input guardrails → routing → hybrid retrieval + rerank → memory → answer → output guardrails, traced |

Until version 0.1.0 the hub kept copies of every component in subfolders and imported them by
editing `sys.path`, with stand-ins when an import failed. Those copies are gone: the SDK installs
the real packages, and a missing one is an explicit error.
