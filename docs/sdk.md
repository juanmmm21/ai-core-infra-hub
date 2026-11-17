# aicore — the AI Core SDK

`aicore` is one pip-installable package that puts the twenty ai-core components behind a single,
typed, stable API. It lives in this repository (`src/aicore`) next to the web panel, which is its
first user.

```bash
pip install "aicore[rag] @ git+https://github.com/juanmmm21/ai-core-infra-hub.git@v0.1.0"
pip install "aicore[agents,ops] @ git+https://github.com/juanmmm21/ai-core-infra-hub.git@v0.1.0"
pip install "aicore[all] @ git+https://github.com/juanmmm21/ai-core-infra-hub.git@v0.1.0"
```

| Extra | Subpackage | Components |
| --- | --- | --- |
| `rag` | `aicore.rag` | bpe-tokenizer-from-scratch, semantic-chunking-engine, multimodal-doc-parser, nano-vector-db, knowledge-graph-extractor, hybrid-search-retrieval-pipeline, cross-encoder-reranker |
| `agents` | `aicore.agents` | agentic-memory-layer, semantic-model-router, llm-guardrails-shield, secure-tool-runtime, orchestra-agents |
| `ops` | `aicore.ops` | llm-inference-server, llm-observability-tracer, llm-eval-harness, synthetic-data-generator, dataset-version-control, llm-annotation-studio |
| `train` | `aicore.train` | llm-qlora-finetuner, contrastive-embedding-trainer (PyTorch) |
| *(none)* | `aicore` | types, embedders, errors — NumPy only |

Importing `aicore` never imports a component. A component is imported the first time its area
is used; if its extra is missing, `aicore.MissingExtraError` names the extra to install.

## How it is built

**The components never import each other** (the ecosystem rule). Each one keeps its own shapes —
nano-vector-db answers dicts with a `distance`, the router a pydantic `RoutingDecision`, the shield
a `(bool, str, str)` tuple. The SDK is the only place where they meet:

```
caller ──► aicore.rag / .agents / .ops / .train      facades: one class per component
               │ speak only aicore.types (frozen dataclasses)
               ▼
           aicore.adapters                           mechanical translation, field-parity checks
               │
               ▼
           components (installed from their repos)   own types, collaborators injected
```

`aicore.adapters` plays the role of `nanosql/translate.py` in Strata: it converts component
results into `aicore.types` and SDK objects into what a component expects (an `Embedder` becomes
the chunker's `EmbeddingProvider`; a nano-vector-db instance becomes the vector store that
hybrid-search and agentic-memory-layer accept by injection). Where a translation depends on a
component's field names, `check_fields` verifies them on first use, so a renamed field fails
there with a clear message instead of as a `KeyError` mid-request.

Collaborators are injected, never imported: hybrid search takes any `VectorStore` and an
`embed_fn`; the memory takes a vector store and an encoder; the annotation studio takes a
tokenizer. One `Embedder` is shared by every area.

## The API

### Types (`aicore`)

`Document`, `Chunk`, `Hit` (higher-is-better `score`, raw per-retriever `signals`), `Answer`,
`Triple`, `MemoryHit`, `RouteDecision`, `GuardVerdict`, `ToolResult`, `Generation`.

### Embedders (`aicore`)

| Class | What | Dependencies |
| --- | --- | --- |
| `HashEmbedder(dimension=512)` | signed feature hashing of words and character trigrams; lexical, deterministic, offline | none |
| `SentenceTransformerEmbedder(model)` | a sentence-transformers model | `sentence-transformers` |
| `OllamaEmbedder(model, base_url)` | Ollama's `/api/embed` | none (standard library HTTP) |

### RAG (`aicore.rag`)

```python
from aicore import Document, SentenceTransformerEmbedder
from aicore.rag import RAGPipeline, Reranker, Chunker
from aicore.ops import InferenceClient

emb = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
rag = RAGPipeline(
    embedder=emb,
    chunker=Chunker(emb, max_tokens=256),
    reranker=Reranker(),                                    # cross-encoder, or lexical offline
    generator=InferenceClient("http://localhost:11434/v1", model="qwen2.5:0.5b"),
)
rag.ingest([Document("guide", open("guide.md").read())])
answer = rag.answer("How do I rotate the keys?")
print(answer.text, [h.id for h in answer.hits])
```

`BPETokenizer`, `Chunker`, `DocumentParser`, `VectorIndex`, `HybridRetriever` (`search`,
`sparse_search`, `dense_search`), `Reranker`, `GraphExtractor`, `RAGPipeline` (`ingest`,
`retrieve`, `answer`, `extractive_answer`), and the `Generator` protocol.

### Agents (`aicore.agents`)

```python
from aicore.agents import Memory, Guard, Router, ToolRuntime

memory = Memory(embedder=emb, path="memory.json", half_life_days=30)
fid = memory.remember("Prefers answers in Spanish", importance=8, metadata={"source": "chat"})
memory.recall("language preference", k=3, where={"source": "chat"})
memory.forget(fid)

verdict = Guard(injection_detector="rules").check_input(user_text)   # GuardVerdict(allowed, text, reason, category)
decision = Router().route(user_text)          # complexity routing to model profiles
intent = IntentRouter(registry_dir="registry").route("sigue", ["active_project"])  # learned intent, champion model
ToolRuntime().run("print(2 + 2)")             # ToolResult(ok, stdout, stderr, error, duration_s)
```

`Memory.remember_many(items, importance=[...], created_at=[...])` bulk-inserts with per-item
importance and dates, embedded in batches. `Guard` takes `injection_detector` (`rules`,
`classifier`, `hybrid`) and `block_destructive`. `IntentRouter` serves the champion of a
semantic-model-router registry (optionally shadowing a challenger).

### Ops (`aicore.ops`)

`InferenceClient` (any OpenAI-compatible server: `chat`, `complete`, `stream`, `models`),
`Tracer` (`with tracer.span(name, inputs) as outputs:`; `Tracer(otlp_endpoint=...)` exports to
OpenTelemetry with the GenAI conventions; `record(name, start, end, parent, **metadata)` adds
spans measured elsewhere), `Evaluator`, `SyntheticData`, `DatasetRepo`, `annotation_app(tokenizer)`.

Evaluation and improvement loops:

* `RAGBenchmark(examples, k).run(name, factory)` — a `RAGPipeline` on HotpotQA with an open
  corpus, scored by llm-eval-harness (`load_hotpotqa`, `load_examples`).
* `RouterImprovementLoop(spec, workdir).run(round, n_per_label, seed, candidates, ...)` —
  synthetic data → dataset versions → training (hashed or LoRA/QLoRA) → harness evaluation →
  champion/challenger promotion. See `pipelines/README.md`.

### Train (`aicore.train`)

`LoraFinetuner(r, alpha, target_modules)`: `apply(model, quantize_4bit=True)`,
`trainable_parameters`, `save_adapters`, `load_adapters`.

## Stability

* The stable API is what `aicore.__all__` and each area's `__all__` export, with the signatures
  documented here. `tests/test_core.py` freezes those lists: changing one is a deliberate act.
* Until 1.0, a minor version may **add** to the API; removals and incompatible changes are
  announced one minor version ahead with a `DeprecationWarning`.
* Anything reached through `.native` (the wrapped component object) or a module starting with
  `_`, and everything in `aicore.adapters`, is internal.
* Each aicore release pins its components to tags. While a roadmap is in progress the extras point
  at branches; the merge order is always: components → their tags → `aicore` pinned to those tags.

## Development

```bash
pip install -e ".[all,dev]"
pytest                                  # SDK + panel
ruff check . && ruff format --check src tests main.py
mypy                                    # --strict on src/aicore
```
