"""AI Core Infra Hub: the web panel over the ai-core components, built on the ``aicore`` SDK.

The panel no longer imports component modules or keeps copies of them: every call goes through
``aicore`` (``pip install -e ".[all]"``), which installs the components from their own
repositories. Configuration, all optional, by environment variable:

* ``AICORE_EMBEDDER``: ``hash`` (default, offline), ``st:<sentence-transformers model>`` or
  ``ollama:<model>``.
* ``AICORE_LLM_BASE_URL`` / ``AICORE_LLM_MODEL`` / ``AICORE_LLM_API_KEY``: any OpenAI-compatible
  server (llm-inference-server, Ollama's ``/v1``, llama.cpp). Without it, answers are extractive.
* ``AICORE_RERANKER=model`` loads the real cross-encoder (needs the component's ``[model]`` extra).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import tempfile
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from importlib import metadata
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from aicore import Chunk, Document, Embedder, HashEmbedder, OllamaEmbedder, SentenceTransformerEmbedder, cosine
from aicore.agents import Guard, Memory, Orchestrator, Router, ToolRuntime
from aicore.ops import DatasetRepo, Evaluator, InferenceClient, SyntheticData, Tracer
from aicore.rag import (
    BPETokenizer,
    Chunker,
    DocumentParser,
    GraphExtractor,
    HybridRetriever,
    RAGPipeline,
    Reranker,
    VectorIndex,
)

logger = logging.getLogger("ai_core_infra_hub")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

HUB_DIR = Path(__file__).resolve().parent
STORAGE = Path(os.environ.get("AICORE_HUB_STORAGE", HUB_DIR / "db_storage"))
UPLOADS = STORAGE / "uploads.jsonl"

# The components whose READMEs seed the panel's own knowledge base (read from the installed
# packages' metadata, so the panel documents exactly the versions it runs).
COMPONENTS = [
    "bpe-tokenizer-from-scratch", "semantic-chunking-engine", "multimodal-doc-parser",
    "contrastive-embedding-trainer", "nano-vector-db", "knowledge-graph-extractor",
    "hybrid-search-retrieval-pipeline", "cross-encoder-reranker", "llm-inference-server",
    "semantic-model-router", "llm-guardrails-shield", "orchestra-agents", "agentic-memory-layer",
    "secure-tool-runtime", "synthetic-data-generator", "dataset-version-control",
    "llm-qlora-finetuner", "llm-eval-harness", "llm-observability-tracer", "llm-annotation-studio",
]  # fmt: skip

BPE_TRAINING_TEXT = (
    "El aprendizaje automático y los embeddings vectoriales son fundamentales para el procesamiento "
    "del lenguaje natural en RAG. Retrieval augmented generation combines search and generation."
)


def make_embedder() -> Embedder:
    spec = os.environ.get("AICORE_EMBEDDER", "hash")
    if spec.startswith("st:"):
        return SentenceTransformerEmbedder(spec[3:])
    if spec.startswith("ollama:"):
        return OllamaEmbedder(spec[7:], base_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    return HashEmbedder(512)


def make_generator() -> InferenceClient | None:
    base = os.environ.get("AICORE_LLM_BASE_URL")
    if not base:
        return None
    return InferenceClient(
        base, model=os.environ.get("AICORE_LLM_MODEL", "default"), api_key=os.environ.get("AICORE_LLM_API_KEY")
    )


def component_readme(dist: str) -> str:
    """A component's README as packaged in its metadata (the long description)."""
    try:
        md = metadata.metadata(dist)
    except metadata.PackageNotFoundError:
        return ""
    body = md.get_payload() if hasattr(md, "get_payload") else None  # type: ignore[attr-defined]
    return str(body or md.get("Description") or "")


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


class Hub:
    """Everything the panel serves, assembled once from the SDK."""

    def __init__(self) -> None:
        STORAGE.mkdir(parents=True, exist_ok=True)
        self.embedder = make_embedder()
        self.generator = make_generator()
        self.retriever = HybridRetriever(self.embedder, index_type="hnsw")
        self.reranker = Reranker(offline=os.environ.get("AICORE_RERANKER") != "model")
        self.rag = RAGPipeline(
            self.embedder, generator=self.generator, reranker=self.reranker, retriever=self.retriever
        )
        self.memory = Memory(self.embedder, path=STORAGE / "memory.json")
        if len(self.memory) == 0:
            self.memory.remember("El usuario prefiere respuestas técnicas, con fórmulas cuando ayudan.", importance=9)
            self.memory.remember("Los componentes de AI Core están escritos desde cero en Python.", importance=8)
        self.guard = Guard()
        self.router = Router()
        self.runtime = ToolRuntime()
        self.tracer = Tracer()
        self.tokenizer = BPETokenizer().train(BPE_TRAINING_TEXT, vocab_size=300)
        self.parser = DocumentParser()
        self.lock = threading.Lock()
        self._seed()

    def _seed(self) -> None:
        chunks: list[Chunk] = []
        for dist in COMPONENTS:
            for i, text in enumerate(paragraphs(component_readme(dist))):
                chunks.append(Chunk(f"{dist}_chunk_{i}", dist, text, i, {"source": f"{dist}/README.md"}))
        readme = HUB_DIR / "README.md"
        if readme.exists():
            for i, text in enumerate(paragraphs(readme.read_text(encoding="utf-8"))):
                chunks.append(
                    Chunk(f"ai-core-infra-hub_chunk_{i}", "ai-core-infra-hub", text, i, {"source": "README.md"})
                )
        self.retriever.add(chunks)
        if UPLOADS.exists():
            for line in UPLOADS.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    self._index(item["prefix"], item["chunks"], item["source"])
        logger.info("Hub ready: %d chunks indexed with %s", len(self.retriever), type(self.embedder).__name__)

    def _index(self, prefix: str, texts: list[str], source: str) -> int:
        chunks = [Chunk(f"{prefix}_chunk_{i}", prefix, t, i, {"source": source}) for i, t in enumerate(texts)]
        fresh = [c for c in chunks if c.id not in self.retriever.index.texts]
        return self.retriever.add(fresh)

    def add_document(self, prefix: str, texts: list[str], source: str) -> int:
        with self.lock:
            added = self._index(prefix, texts, source)
            with UPLOADS.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"prefix": prefix, "chunks": texts, "source": source}, ensure_ascii=False) + "\n")
        return added


hub: Hub | None = None


def get_hub() -> Hub:
    global hub
    if hub is None:
        hub = Hub()
    return hub


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    get_hub()
    yield


app = FastAPI(
    title="AI Core Infra Hub",
    description="Panel over the ai-core components, built on the aicore SDK.",
    lifespan=lifespan,
)


class DocumentUploadRequest(BaseModel):
    title: str
    content: str


class TokenizeRequest(BaseModel):
    text: str


class ChunkRequest(BaseModel):
    text: str
    threshold: float = 1.2


class ParseRequest(BaseModel):
    filename: str
    content: str


class SimilarityRequest(BaseModel):
    text1: str
    text2: str


class VectorInsertRequest(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorQueryRequest(BaseModel):
    query: str
    top_k: int = 3
    filter: dict[str, Any] | None = None


class KGExtractRequest(BaseModel):
    text: str


class HybridSearchRequest(BaseModel):
    query: str
    alpha: float = 0.5


class RerankRequest(BaseModel):
    query: str
    documents: list[str]


class RouterRequest(BaseModel):
    prompt: str


class GuardrailsRequest(BaseModel):
    prompt: str
    response_to_check: str | None = None
    context: str | None = None


class AgentRequest(BaseModel):
    task: str


class MemorySaveRequest(BaseModel):
    fact: str
    importance: int = 5


class MemoryRecallRequest(BaseModel):
    query: str
    decay_factor: float = 0.05
    time_elapsed: float = 0.0


class RunCodeRequest(BaseModel):
    code: str


class SyntheticRequest(BaseModel):
    topics: list[str]
    count: int = 3


class DVCCommitRequest(BaseModel):
    filename: str
    data: list[dict[str, Any]]
    message: str


class DVCDiffRequest(BaseModel):
    commit_a: str
    commit_b: str


class QLoraTrainRequest(BaseModel):
    r: int = 8
    alpha: int = 16
    epochs: int = 5


class BenchmarkRequest(BaseModel):
    candidate: str
    reference: str


class UnifiedPipelineRequest(BaseModel):
    prompt: str
    context: str | None = None


@app.get("/", response_class=HTMLResponse)
async def get_index() -> str:
    template = HUB_DIR / "templates" / "index.html"
    if not template.exists():
        raise HTTPException(status_code=404, detail="templates/index.html not found")
    return template.read_text(encoding="utf-8")


# --- RAG -------------------------------------------------------------------------------------


@app.post("/api/bpe/tokenize")
def api_bpe_tokenize(req: TokenizeRequest) -> dict[str, Any]:
    tok = get_hub().tokenizer
    ids = tok.encode(req.text)
    return {"tokens": ids, "splits": tok.pieces(req.text), "reconstructed": tok.decode(ids)}


@app.post("/api/chunker/chunk")
def api_chunker_chunk(req: ChunkRequest) -> dict[str, Any]:
    h = get_hub()
    chunks = Chunker(h.embedder, tokenizer=h.tokenizer, threshold=req.threshold).split(Document("input", req.text))
    return {"chunks": [c.text for c in chunks], "tokens": [c.metadata.get("tokens") for c in chunks]}


@app.post("/api/parser/parse")
def api_parser_parse(req: ParseRequest) -> dict[str, Any]:
    suffix = Path(req.filename).suffix or ".txt"
    if suffix.lower() == ".pdf":
        raise HTTPException(status_code=400, detail="Upload PDFs through /api/documents/upload-file")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"input{suffix}"
        path.write_text(req.content, encoding="utf-8")
        doc = get_hub().parser.parse(path, document_id=Path(req.filename).name)
    return {"markdown": f"# {doc.id}\n\n{doc.text}"}


@app.post("/api/contrastive/similarity")
def api_contrastive_similarity(req: SimilarityRequest) -> dict[str, Any]:
    h = get_hub()
    v1, v2 = h.embedder.embed([req.text1, req.text2])
    return {"similarity": cosine(v1, v2), "embedder": type(h.embedder).__name__}


PLAYGROUND = [
    ("doc1", "Bases de datos vectoriales HNSW", {"category": "database"}),
    ("doc2", "Procesamiento de embeddings semanticos", {"category": "embeddings"}),
    ("doc3", "Agentes y planificadores autonomos ReAct", {"category": "agents"}),
]
_playground: VectorIndex | None = None


def playground() -> VectorIndex:
    global _playground
    if _playground is None:
        h = get_hub()
        _playground = VectorIndex(h.embedder.dimension, index_type="hnsw")
        for doc_id, text, meta in PLAYGROUND:
            _playground.add(doc_id, h.embedder.embed([text])[0], text, meta)
    return _playground


@app.post("/api/vector-db/insert")
def api_vector_db_insert(req: VectorInsertRequest) -> dict[str, Any]:
    index = playground()
    index.add(req.id, get_hub().embedder.embed([req.text])[0], req.text, req.metadata)
    return {"status": "success", "id": req.id, "size": len(index)}


@app.post("/api/vector-db/query")
def api_vector_db_query(req: VectorQueryRequest) -> dict[str, Any]:
    hits = playground().search(get_hub().embedder.embed([req.query])[0], k=req.top_k, where=req.filter)
    return {
        "results": [
            {"id": h.id, "distance": h.signals.get("dense_distance"), "score": h.score, "metadata": dict(h.metadata)}
            for h in hits
        ]
    }


@app.get("/api/vector-db/list")
def api_vector_db_list() -> dict[str, Any]:
    index = get_hub().retriever.index
    records = [
        {
            "id": str(doc_id),
            "metadata": index.db.metadata.get(doc_id, {}),
            "text": index.texts.get(str(doc_id), ""),
            "vector_preview": [round(float(x), 4) for x in list(vec)[:5]],
        }
        for doc_id, vec in index.db.vectors.items()
    ]
    return {"status": "success", "records": records}


@app.post("/api/knowledge-graph/extract")
def api_kg_extract(req: KGExtractRequest) -> dict[str, Any]:
    triples = GraphExtractor().extract(req.text)
    names: dict[str, str] = {}
    for t in triples:
        names.setdefault(t.subject, t.subject_type or "entity")
        names.setdefault(t.object, t.object_type or "entity")
    nodes = [{"id": n, "type": kind, "val": 10} for n, kind in names.items()]
    links = [{"source": t.subject, "target": t.object, "label": t.relation, "value": 1} for t in triples]

    def q(s: str) -> str:
        return s.replace("\\", "\\\\").replace("'", "\\'")

    cypher = [f"MERGE (:Entity {{id: '{q(n)}', type: '{q(k)}'}})" for n, k in names.items()]
    rel = [re.sub(r"\W+", "_", t.relation).upper() or "RELATED_TO" for t in triples]
    cypher += [
        f"MATCH (a:Entity {{id: '{q(t.subject)}'}}), (b:Entity {{id: '{q(t.object)}'}}) MERGE (a)-[:{r}]->(b)"
        for t, r in zip(triples, rel, strict=True)
    ]
    return {"nodes": nodes, "links": links, "cypher": "\n".join(cypher)}


HYBRID_DEMO = {
    "doc1": "La arquitectura RAG combina busqueda clasica BM25 y busqueda vectorial semantica HNSW.",
    "doc2": "Los adaptadores LoRA e inferencia en NormalFloat4 optimizan el consumo de VRAM en PyTorch.",
    "doc3": "La capa de memoria de largo plazo episódica simula la curva del olvido de Ebbinghaus temporal.",
}


@app.post("/api/hybrid-search/search")
def api_hybrid_search(req: HybridSearchRequest) -> dict[str, Any]:
    retriever = HybridRetriever(get_hub().embedder)
    retriever.add(Chunk(i, i, t, 0) for i, t in HYBRID_DEMO.items())

    def view(hits: list[Any]) -> list[dict[str, Any]]:
        return [{"id": h.id, "score": h.score, **dict(h.signals)} for h in hits]

    return {
        "sparse": view(retriever.sparse_search(req.query, k=3)),
        "dense": view(retriever.dense_search(req.query, k=3)),
        "rrf": view(retriever.search(req.query, k=3, fusion="rrf")),
        "score_fusion": view(retriever.search(req.query, k=3, fusion="score", alpha=req.alpha)),
    }


@app.post("/api/reranker/rerank")
def api_reranker_rerank(req: RerankRequest) -> dict[str, Any]:
    from aicore import Hit

    hits = [Hit(str(i), d, 0.0) for i, d in enumerate(req.documents)]
    reranked = get_hub().reranker.rerank(req.query, hits, k=len(hits))
    return {"results": [{"id": int(h.id), "text": h.text, "score": h.score} for h in reranked]}


# --- Inference -------------------------------------------------------------------------------


def require_generator() -> InferenceClient:
    gen = get_hub().generator
    if gen is None:
        raise HTTPException(
            status_code=503,
            detail="No inference backend: set AICORE_LLM_BASE_URL to an OpenAI-compatible server "
            "(llm-inference-server, Ollama /v1, llama.cpp).",
        )
    return gen


@app.post("/api/inference/generate")
def api_inference_generate(req: RouterRequest) -> dict[str, Any]:
    gen = require_generator().chat([{"role": "user", "content": req.prompt}], max_tokens=256)
    return {
        "text": gen.text,
        "model": gen.model,
        "latency_ms": round(gen.latency_s * 1000, 1),
        "prompt_tokens": gen.prompt_tokens,
        "completion_tokens": gen.completion_tokens,
    }


@app.post("/api/inference/generate_stream")
def api_inference_stream(req: RouterRequest) -> StreamingResponse:
    gen = require_generator()

    def events() -> Iterator[str]:
        yield "data: [START]\n\n"
        try:
            for delta in gen.stream([{"role": "user", "content": req.prompt}], max_tokens=256):
                yield f"data: {delta}\n\n"
        except Exception as exc:  # the stream has started: report in-band
            yield f"data: [ERROR] {exc}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


# --- Agents ----------------------------------------------------------------------------------


@app.post("/api/router/route")
def api_router_route(req: RouterRequest) -> dict[str, Any]:
    d = get_hub().router.route(req.prompt)
    return {
        "selected_model": d.label,
        "reason": d.reason,
        "estimated_cost": d.estimated_cost_per_1k_tokens,
        "confidence": d.confidence,
        "path": d.path,
    }


@app.post("/api/guardrails/check")
def api_guardrails_check(req: GuardrailsRequest) -> dict[str, Any]:
    guard = get_hub().guard
    inp = guard.check_input(req.prompt)
    out = guard.check_output(req.response_to_check, req.context) if req.response_to_check else None
    return {
        "input": {"safe": inp.allowed, "processed": inp.text, "reason": inp.reason, "category": inp.category},
        "output": {
            "safe": out.allowed if out else True,
            "processed": out.text if out else "",
            "reason": out.reason if out else "",
            "category": out.category if out else "",
        },
    }


@app.post("/api/agents/run")
def api_agents_run(req: AgentRequest) -> dict[str, Any]:
    h = get_hub()
    orch = Orchestrator(name="HubSupervisor")

    def search_db(query: str) -> str:
        hits = h.retriever.search(query, k=2)
        return " | ".join(hit.text[:200] for hit in hits) or "sin resultados"

    orch.add_agent(
        "ResearcherAgent", "Investigador", "Busca datos en la base de conocimiento.", {"search_db": search_db}
    )
    orch.add_agent("WriterAgent", "Redactor", "Redacta informes técnicos.", {"write_report": lambda t: f"INFORME: {t}"})
    return {"plan": orch.plan(req.task), "result": orch.run(req.task)}


@app.post("/api/memory/save")
def api_memory_save(req: MemorySaveRequest) -> dict[str, Any]:
    fact_id = get_hub().memory.remember(req.fact, importance=req.importance)
    return {"status": "success", "id": fact_id, "saved_fact": req.fact}


@app.post("/api/memory/recall")
def api_memory_recall(req: MemoryRecallRequest) -> dict[str, Any]:
    h = get_hub()
    now = time.time()
    stored = [vars_of(m) for m in h.memory.recall(req.query, k=3, now=now + req.time_elapsed)]
    # The decay demo: one fact created `time_elapsed` seconds ago, forgotten at rate λ.
    half_life_days = (math.log(2) / req.decay_factor) / 86400.0 if req.decay_factor > 0 else None
    demo = Memory(h.embedder, half_life_days=half_life_days)
    demo.remember("Al usuario le gusta programar en Python", importance=8, created_at=now - req.time_elapsed)
    return {"results": [vars_of(m) for m in demo.recall(req.query, k=1, now=now)], "stored": stored}


def vars_of(hit: Any) -> dict[str, Any]:
    return {
        "id": hit.id,
        "fact": hit.text,
        "recall_score": hit.score,
        "importance": hit.importance,
        "created_at": hit.created_at,
    }


@app.post("/api/runtime/run")
def api_runtime_run(req: RunCodeRequest) -> dict[str, Any]:
    r = get_hub().runtime.run(req.code, timeout_s=1.5)
    return {"success": r.ok, "stdout": r.stdout, "stderr": r.stderr, "error": r.error, "duration_s": r.duration_s}


# --- Ops -------------------------------------------------------------------------------------


@app.post("/api/synthetic/generate")
def api_synthetic_generate(req: SyntheticRequest) -> dict[str, Any]:
    gen = SyntheticData(min_length=10, dedup_threshold=0.5)
    return {
        "instructions": [
            {"prompt": r["instruction"], "output": r["output"]} for r in gen.instructions(req.topics, req.count)
        ],
        "dpo": gen.preferences(req.topics, req.count),
    }


def dataset_repo() -> DatasetRepo:
    return DatasetRepo(STORAGE / "dvc_store")


@app.post("/api/dvc/commit")
def api_dvc_commit(req: DVCCommitRequest) -> dict[str, Any]:
    datasets = STORAGE / "datasets"
    datasets.mkdir(parents=True, exist_ok=True)
    path = datasets / (Path(req.filename).name or "dataset.json")  # never write outside storage
    path.write_text(json.dumps(req.data, ensure_ascii=False), encoding="utf-8")
    repo = dataset_repo()
    return {"commit_hash": repo.commit(path, req.message), "log": repo.log()}


@app.post("/api/dvc/diff")
def api_dvc_diff(req: DVCDiffRequest) -> dict[str, Any]:
    return dataset_repo().diff(req.commit_a, req.commit_b)


@app.get("/api/dvc/log")
def api_dvc_log() -> list[dict[str, Any]]:
    return dataset_repo().log()


@app.post("/api/qlora/train")
def api_qlora_train(req: QLoraTrainRequest) -> dict[str, Any]:
    """A real, tiny LoRA run: a frozen (NF4-quantized) 2-layer network learns a new linear map
    through rank-``r`` adapters only. Needs ``aicore[train]`` (PyTorch)."""
    try:
        import torch

        from aicore.train import LoraFinetuner
    except ImportError as exc:
        raise HTTPException(
            status_code=503, detail=f'Training needs PyTorch: pip install "aicore[train]" ({exc})'
        ) from exc

    torch.manual_seed(0)
    model = torch.nn.Sequential()
    model.add_module("fc1", torch.nn.Linear(32, 64))
    model.add_module("act", torch.nn.ReLU())
    model.add_module("fc2", torch.nn.Linear(64, 8))
    x = torch.randn(256, 32)
    y = x @ torch.randn(32, 8) * 0.5
    ft = LoraFinetuner(r=req.r, alpha=req.alpha, target_modules=["fc1", "fc2"], dropout=0.0)
    model = ft.apply(model, quantize_4bit=True)
    params = [p for _, p in ft.trainable_parameters(model)]
    opt = torch.optim.Adam(params, lr=1e-2)
    history = []
    for epoch in range(1, max(1, min(req.epochs, 200)) + 1):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        history.append({"epoch": epoch, "loss": float(loss.detach())})
    shapes = {name: list(p.shape) for name, p in ft.trainable_parameters(model)}
    return {"status": "success", "rank": req.r, "alpha": req.alpha, "loss_history": history, "saved_adapters": shapes}


@app.post("/api/eval/benchmark")
def api_eval_benchmark(req: BenchmarkRequest) -> dict[str, Any]:
    return Evaluator().score(req.candidate, req.reference)


@app.get("/api/telemetry")
def api_telemetry() -> list[dict[str, Any]]:
    return get_hub().tracer.tree()


# --- Documents -------------------------------------------------------------------------------

TEXT_EXTENSIONS = {".txt", ".md", ".json", ".py", ".js", ".csv", ".yaml", ".yml", ".ini"}


def slug(text: str) -> str:
    return re.sub(r"[^\w.-]+", "_", text.strip().lower())


@app.post("/api/documents/upload-file")
async def api_documents_upload_file(file: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008 (FastAPI)
    h = get_hub()
    filename = Path(file.filename or "upload").name
    ext = Path(filename).suffix.lower()
    raw = await file.read()
    if ext == ".pdf":
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "upload.pdf"
            path.write_bytes(raw)
            text = h.parser.parse(path, document_id=filename).text
    elif ext in TEXT_EXTENSIONS:
        text = raw.decode("utf-8", errors="ignore")
    else:
        return {"error": f"Formato '{ext}' no soportado. Sube un PDF, TXT, MD, etc."}
    chunks = paragraphs(text)
    if not chunks:
        return {"error": "El archivo está vacío o no contiene texto extraíble."}
    await asyncio.to_thread(h.add_document, f"file_{slug(filename)}", chunks, filename)
    return {"status": "success", "filename": filename, "chunks_count": len(chunks), "total_chars": len(text)}


@app.post("/api/documents/upload")
def api_documents_upload(req: DocumentUploadRequest) -> dict[str, Any]:
    chunks = paragraphs(req.content) or [req.content.strip()]
    get_hub().add_document(f"uploaded_{slug(req.title)}", chunks, req.title)
    return {"status": "success", "chunks_count": len(chunks)}


# --- The whole pipeline ----------------------------------------------------------------------


def source_name(chunk_id: str) -> str:
    return re.split(r"_chunk_|#", chunk_id)[0].removeprefix("file_").removeprefix("uploaded_")


@app.post("/api/pipeline/run")
def api_pipeline_run(req: UnifiedPipelineRequest) -> dict[str, Any]:
    h = get_hub()
    tr = h.tracer
    tr.clear()
    with tr.span("AI Core pipeline", inputs={"prompt": req.prompt}) as root:
        with tr.span("1. Input guardrails (llm-guardrails-shield)") as out:
            verdict = h.guard.check_input(req.prompt)
            out.update(allowed=verdict.allowed, category=verdict.category)
        if not verdict.allowed:
            root["blocked"] = verdict.reason
            return {"status": "blocked", "reason": verdict.reason, "telemetry": tr.tree()}
        prompt = verdict.text

        with tr.span("2. Model routing (semantic-model-router)") as out:
            decision = h.router.route(prompt)
            out.update(model=decision.label, path=decision.path)

        with tr.span("3. Hybrid retrieval + rerank (hybrid-search, nano-vector-db, cross-encoder)") as out:
            hits = h.rag.retrieve(prompt, k=3)
            out["hits"] = [{"id": x.id, "score": round(x.score, 4)} for x in hits]

        with tr.span("4. Memory recall (agentic-memory-layer)") as out:
            memories = h.memory.recall(prompt, k=2)
            out["facts"] = [m.text for m in memories]

        with tr.span("5. Answer generation") as out:
            if h.generator is not None:
                answer = h.rag.answer(prompt, k=3).text
                out["backend"] = h.generator.model
            else:
                answer = h.rag.extractive_answer(prompt, hits)
                out["backend"] = "extractive"
            sources = sorted({source_name(x.id) for x in hits})
            response = f"{answer}\n\nFuentes: {', '.join(sources)}" if answer else "No he encontrado nada relacionado."

        with tr.span("6. Output guardrails (llm-guardrails-shield)") as out:
            context = "\n\n".join(x.text for x in hits)
            checked = h.guard.check_output(response, context)
            out.update(allowed=checked.allowed, category=checked.category)

    if not checked.allowed:
        return {"status": "blocked", "reason": checked.reason, "telemetry": tr.tree()}
    return {
        "status": "success",
        "model": decision.label if h.generator is None else h.generator.model,
        "cost": decision.estimated_cost_per_1k_tokens,
        "response": checked.text,
        "sources": [{"id": x.id, "score": x.score, "source": x.metadata.get("source")} for x in hits],
        "memories": [m.text for m in memories],
        "telemetry": tr.tree(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000)
