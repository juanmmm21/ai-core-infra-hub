import os
import sys
import time
import math
import uuid
import json
import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import asyncio

# Setup path mapping to sibling projects (Interlinking)
HUB_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(HUB_DIR)

PROJECT_PATHS = [
    "bpe-tokenizer-from-scratch",
    "semantic-chunking-engine",
    "multimodal-doc-parser",
    "llm-annotation-studio",
    "contrastive-embedding-trainer",
    "nano-vector-db",
    "knowledge-graph-extractor",
    "hybrid-search-retrieval-pipeline",
    "cross-encoder-reranker",
    "llm-inference-server",
    "semantic-model-router",
    "llm-guardrails-shield",
    "orchestra-agents",
    "agentic-memory-layer",
    "secure-tool-runtime",
    "synthetic-data-generator",
    "dataset-version-control",
    "llm-qlora-finetuner",
    "llm-eval-harness",
    "llm-observability-tracer",
    "nexus-second-brain"
]

for p in PROJECT_PATHS:
    p_path = os.path.join(PARENT_DIR, p)
    if p_path not in sys.path:
        sys.path.append(p_path)

# Safe imports with placeholders if necessary
try:
    from tokenizer import BPETokenizer
except ImportError:
    BPETokenizer = None

try:
    from chunker import SemanticChunker
    from embedding_provider import MockEmbeddingProvider
except ImportError:
    SemanticChunker, MockEmbeddingProvider = None, None

try:
    from parser import MultimodalDocParser
    from backend import LocalExtractBackend
except ImportError:
    MultimodalDocParser, LocalExtractBackend = None, None

try:
    from database import NanoVectorDB
except ImportError:
    NanoVectorDB = None

try:
    from extractor import KnowledgeGraphStore
except ImportError:
    KnowledgeGraphStore = None

try:
    from pipeline import HybridSearchPipeline
    from bm25 import BM25Retriever
except ImportError:
    HybridSearchPipeline, BM25Retriever = None, None

try:
    from reranker import CrossEncoderReranker
except ImportError:
    CrossEncoderReranker = None

try:
    from inference_engine import InferenceEngine
except ImportError:
    InferenceEngine = None

try:
    from router import SemanticModelRouter, ModelProfile
except ImportError:
    SemanticModelRouter, ModelProfile = None, None

try:
    from shield import LLMGuardrailsShield
except ImportError:
    LLMGuardrailsShield = None

try:
    from agent import AutonomousAgent
    from orchestrator import AgentOrchestrator
except ImportError:
    AutonomousAgent, AgentOrchestrator = None, None

try:
    from memory import AgenticMemory
except ImportError:
    AgenticMemory = None

try:
    from runtime import SecureToolRuntime
except ImportError:
    SecureToolRuntime = None

try:
    from generator import SyntheticDataGenerator
except ImportError:
    SyntheticDataGenerator = None

try:
    from dvc import DatasetVCS
except ImportError:
    DatasetVCS = None

try:
    from finetuner import NF4Quantizer, LoraConfig
except ImportError:
    NF4Quantizer, LoraConfig = None, None

try:
    from evaluator import LLMEvaluator
except ImportError:
    LLMEvaluator = None

try:
    from tracer import GlobalTracer
except ImportError:
    # Fallback tracer
    class GlobalTracer:
        def __init__(self):
            self.root_spans = []
        def start_span(self, name, inputs=None, metadata=None):
            span = {"name": name, "start_time": time.time(), "end_time": None, "inputs": inputs or {}, "metadata": metadata or {}, "children": []}
            self.root_spans.append(span)
            return span
        def end_span(self, outputs=None, error=None, metadata_update=None):
            if self.root_spans:
                span = self.root_spans[-1]
                span["end_time"] = time.time()
                if outputs: span["outputs"] = outputs
                if metadata_update: span["metadata"].update(metadata_update)
        def clear(self):
            self.root_spans = []
        def get_trace_tree(self):
            return self.root_spans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_core_infra_hub")

app = FastAPI(
    title="AI Core Infra Hub",
    description="Panel unificado de demostración y orquestación para los 21 subproyectos de la infraestructura de IA."
)

# Global instances initialization
global_tracer = GlobalTracer()

# --- MODELOS DE DATOS ---
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
    metadata: Dict[str, Any] = Field(default_factory=dict)

class VectorQueryRequest(BaseModel):
    query: str
    top_k: int = 3
    filter: Optional[Dict[str, Any]] = None

class KGExtractRequest(BaseModel):
    text: str

class HybridSearchRequest(BaseModel):
    query: str
    alpha: float = 0.5

class RerankRequest(BaseModel):
    query: str
    documents: List[str]

class RouterRequest(BaseModel):
    prompt: str

class GuardrailsRequest(BaseModel):
    prompt: str
    context: Optional[str] = None
    response_to_check: Optional[str] = None

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
    topics: List[str]
    count: int = 3

class DVCCommitRequest(BaseModel):
    filename: str
    data: List[Dict[str, Any]]
    message: str

class QLoraTrainRequest(BaseModel):
    r: int = 8
    alpha: float = 16.0
    epochs: int = 3

class BenchmarkRequest(BaseModel):
    candidate: str
    reference: str

class UnifiedPipelineRequest(BaseModel):
    prompt: str
    context: Optional[str] = None

# --- ENDPOINTS ---

@app.get("/", response_class=HTMLResponse)
async def get_index():
    template_path = os.path.join(HUB_DIR, "templates", "index.html")
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="Plantilla index.html no encontrada.")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()

# 1. BPE Tokenizer
@app.post("/api/bpe/tokenize")
def api_bpe_tokenize(req: TokenizeRequest):
    if not BPETokenizer:
        return {"error": "BPE Tokenizer no importado."}
    # Creamos un tokenizador temporal y lo entrenamos con un texto base o cargamos existente
    tokenizer = BPETokenizer()
    base_text = "El aprendizaje automático y los embeddings vectoriales son fundamentales para el procesamiento del lenguaje natural en RAG."
    tokenizer.train(base_text, vocab_size=300)
    tokens = tokenizer.encode(req.text)
    decoded = [tokenizer.decode([t]) for t in tokens]
    return {
        "tokens": tokens,
        "splits": decoded,
        "reconstructed": tokenizer.decode(tokens)
    }

# 2. Semantic Chunker
@app.post("/api/chunker/chunk")
def api_chunker_chunk(req: ChunkRequest):
    if not SemanticChunker or not MockEmbeddingProvider:
        return {"error": "Semantic Chunker o MockEmbeddingProvider no importados."}
    provider = MockEmbeddingProvider()
    chunker = SemanticChunker(embedding_provider=provider, threshold_factor=req.threshold)
    chunks_raw = chunker.chunk_text(req.text)
    chunks = [c["text"] for c in chunks_raw]
    return {"chunks": chunks}


# 3. Multimodal Parser
@app.post("/api/parser/parse")
def api_parser_parse(req: ParseRequest):
    # Simulamos parsear guardando un archivo ficticio y llamando LocalExtractBackend
    if not MultimodalDocParser or not LocalExtractBackend:
        return {"error": "Multimodal Parser o LocalExtractBackend no importados."}
    temp_file = "temp_parsing.pdf"
    # Dado que no es un PDF real, simulamos el Markdown retornado
    markdown = f"# Documento Extraido: {req.filename}\n\n{req.content}\n\n*Nota: Procesado mediante LocalExtractBackend.*"
    return {"markdown": markdown}

# 5. Contrastive Similarity
@app.post("/api/contrastive/similarity")
def api_contrastive_similarity(req: SimilarityRequest):
    # Usamos MockEmbeddingProvider para simular la codificación y calculamos coseno
    if not MockEmbeddingProvider:
        return {"error": "MockEmbeddingProvider no importado."}
    provider = MockEmbeddingProvider()
    v1 = np.array(provider.get_embeddings([req.text1])[0])
    v2 = np.array(provider.get_embeddings([req.text2])[0])
    sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return {"similarity": sim}

# 6. Nano Vector DB
@app.post("/api/vector-db/insert")
def api_vector_db_insert(req: VectorInsertRequest):
    if not NanoVectorDB or not MockEmbeddingProvider:
        return {"error": "NanoVectorDB o MockEmbeddingProvider no importados."}
    db = NanoVectorDB(dimension=64, index_type="hnsw")
    provider = MockEmbeddingProvider()
    vec = provider.get_embeddings([req.text])[0]
    # Ajustamos a dim 64
    vec_64 = vec[:64] if len(vec) >= 64 else vec + [0.0]*(64-len(vec))
    db.insert(id=req.id, vector=vec_64, metadata=req.metadata)
    return {"status": "success", "id": req.id}

@app.post("/api/vector-db/query")
def api_vector_db_query(req: VectorQueryRequest):
    if not NanoVectorDB or not MockEmbeddingProvider:
        return {"error": "NanoVectorDB o MockEmbeddingProvider no importados."}
    db = NanoVectorDB(dimension=64, index_type="hnsw")
    provider = MockEmbeddingProvider()
    # Insertamos unos vectores mock de prueba
    docs = [
        ("doc1", "Bases de datos vectoriales HNSW", {"category": "database"}),
        ("doc2", "Procesamiento de embeddings semanticos", {"category": "embeddings"}),
        ("doc3", "Agentes y planificadores autonomos ReAct", {"category": "agents"})
    ]
    for doc_id, text, meta in docs:
        vec = provider.get_embeddings([text])[0]
        vec_64 = vec[:64] if len(vec) >= 64 else vec + [0.0]*(64-len(vec))
        db.insert(id=doc_id, vector=vec_64, metadata=meta)
        
    query_vec = provider.get_embeddings([req.query])[0]
    query_64 = query_vec[:64] if len(query_vec) >= 64 else query_vec + [0.0]*(64-len(query_vec))
    results = db.query(vector=query_64, top_k=req.top_k, filter=req.filter)
    # Formateamos
    formatted = []
    for r in results:
        formatted.append({"id": r["id"], "distance": float(r["distance"]), "metadata": r["metadata"]})
    return {"results": formatted}

# 7. Knowledge Graph Extractor
@app.post("/api/knowledge-graph/extract")
def api_kg_extract(req: KGExtractRequest):
    # Simulador offline del extractor semantico de relaciones
    text_lower = req.text.lower()
    nodes = [{"id": "UserText", "type": "root", "val": 15}]
    links = []
    
    concepts = {
        "rag": "RAG Pipeline",
        "lora": "LoRA Adapter",
        "vector": "Vector Database",
        "dvc": "Version Control"
    }
    
    for key, name in concepts.items():
        if key in text_lower:
            nodes.append({"id": name, "type": "concept", "val": 10})
            links.append({"source": "UserText", "target": name, "value": 1})
            
    cypher_queries = [
        f"MERGE (n:Root {{id: 'UserText'}})"
    ]
    for n in nodes[1:]:
        cypher_queries.append(f"MERGE (c:Concept {{id: '{n['id']}'}})")
        cypher_queries.append(f"MERGE (n:Root {{id: 'UserText'}})-[:CONTAINS]->(c:Concept {{id: '{n['id']}'}})")
        
    return {
        "nodes": nodes,
        "links": links,
        "cypher": "\n".join(cypher_queries)
    }

# 8. Hybrid Search Pipeline
@app.post("/api/hybrid-search/search")
def api_hybrid_search(req: HybridSearchRequest):
    if not NanoVectorDB or not BM25Retriever:
        return {"error": "NanoVectorDB o BM25Retriever no importados."}
    
    # Corpus de prueba
    corpus = {
        "doc1": "La arquitectura RAG combina busqueda clasica BM25 y busqueda vectorial semantica HNSW.",
        "doc2": "Los adaptadores LoRA e inferencia en NormalFloat4 optimizan el consumo de VRAM en PyTorch.",
        "doc3": "La capa de memoria de largo plazo episódica simula la curva del olvido de Ebbinghaus temporal."
    }
    
    # 1. Sparse BM25
    bm25 = BM25Retriever()
    bm25.fit(corpus)
    sparse_res = bm25.retrieve(req.query, top_k=3)
    
    # 2. Dense DB
    db = NanoVectorDB(dimension=16, index_type="hnsw")
    # Generador hash local
    for doc_id, text in corpus.items():
        # Generar hash vector de 16 dim
        vec = [float(abs(hash(text + str(i))) % 100) / 100.0 for i in range(16)]
        db.insert(id=doc_id, vector=vec, metadata={"text": text})
        
    q_vec = [float(abs(hash(req.query + str(i))) % 100) / 100.0 for i in range(16)]
    dense_raw = db.query(q_vec, top_k=3)
    dense_res = [(r["distance"], r["id"]) for r in dense_raw]
    
    # 3. Fusion
    rrf_res = reciprocal_rank_fusion(dense_res, sparse_res, k=10)
    score_res = score_normalization_fusion(dense_res, sparse_res, alpha=req.alpha, metric="cosine")
    
    return {
        "sparse": [{"id": doc_id, "score": float(score)} for score, doc_id in sparse_res],
        "dense": [{"id": r["id"], "distance": float(r["distance"])} for r in dense_raw],
        "rrf": [{"id": doc_id, "score": float(score)} for score, doc_id in rrf_res],
        "score_fusion": [{"id": doc_id, "score": float(score)} for score, doc_id in score_res]
    }

# 9. Cross Encoder Reranker
@app.post("/api/reranker/rerank")
def api_reranker_rerank(req: RerankRequest):
    if not CrossEncoderReranker:
        return {"error": "CrossEncoderReranker no importado."}
    # Usamos modo offline para evitar descargar modelo de HF
    reranker = CrossEncoderReranker(model_name="dummy", device="cpu")
    reranker.is_online = False
    
    docs_dict = [{"id": i, "text": doc} for i, doc in enumerate(req.documents)]
    reranked = reranker.rerank(req.query, docs_dict, top_k=len(req.documents))
    
    return {
        "results": [{"id": r["id"], "text": r["text"], "score": float(r["rerank_score"])} for r in reranked]
    }

# 10. LLM Inference Server (Dynamic Batching & Streaming)
@app.post("/api/inference/generate")
async def api_inference_generate(req: RouterRequest):
    # Simulamos el dynamic batching con un pequeño retardo
    await asyncio.sleep(0.06)
    # Inferencia simulada
    text = f"[Inferencia Procesada] Respuesta al prompt: '{req.prompt}'. Procesado con Dynamic Batching."
    return {"text": text, "latency_ms": 60.0, "batch_size": 1}

@app.post("/api/inference/generate_stream")
def api_inference_stream(req: RouterRequest):
    async def sse_generator():
        # Emitir tokens simulados
        yield "data: [START]\n\n"
        words = f"Inferencia en streaming simulado para el prompt: {req.prompt}. Completado exitosamente.".split(" ")
        for word in words:
            yield f"data: {word} \n\n"
            await asyncio.sleep(0.04)
        yield "data: [DONE]\n\n"
    return StreamingResponse(sse_generator(), media_type="text/event-stream")

# 11. Semantic Router
@app.post("/api/router/route")
def api_router_route(req: RouterRequest):
    if not SemanticModelRouter:
        return {"error": "SemanticModelRouter no importado."}
    router = SemanticModelRouter()
    decision = router.route(req.prompt)
    return {
        "selected_model": decision.selected_model,
        "reason": decision.reason,
        "estimated_cost": decision.estimated_cost_per_1k_tokens,
        "complexity": decision.complexity_category,
        "path": decision.routing_path
    }

# 12. LLM Guardrails Shield
@app.post("/api/guardrails/check")
def api_guardrails_check(req: GuardrailsRequest):
    if not LLMGuardrailsShield:
        return {"error": "LLMGuardrailsShield no importado."}
    shield = LLMGuardrailsShield()
    
    input_safe, clean_prompt, input_reason = shield.validate_input(req.prompt)
    
    output_safe = True
    clean_output = ""
    output_reason = ""
    if req.response_to_check:
        output_safe, clean_output, output_reason = shield.validate_output(req.response_to_check, req.context)
        
    return {
        "input": {
            "safe": input_safe,
            "processed": clean_prompt,
            "reason": input_reason
        },
        "output": {
            "safe": output_safe,
            "processed": clean_output,
            "reason": output_reason
        }
    }

# 13. Orchestra Agents
@app.post("/api/agents/run")
def api_agents_run(req: AgentRequest):
    if not AutonomousAgent or not AgentOrchestrator:
        return {"error": "AutonomousAgent o AgentOrchestrator no importados."}
        
    async def run_agents():
        orchestrator = AgentOrchestrator(name="HubSupervisor")
        researcher = AutonomousAgent("ResearcherAgent", "Investigador", "Busca datos en la DB.", use_mock_llm=True)
        writer = AutonomousAgent("WriterAgent", "Redactor", "Redacta informes tecnicos.", use_mock_llm=True)
        
        # Registrar herramientas mock
        researcher.add_tool("search_db", lambda q: f"Especificaciones de {q}: HNSW, similitud coseno, in-memory.")
        writer.add_tool("write_report", lambda t: f"INFORME FINAL: {t}")
        
        orchestrator.register_agent(researcher)
        orchestrator.register_agent(writer)
        
        result = orchestrator.execute(req.task)
        
        # Logs de los scratchpads
        logs = []
        for name, agent in orchestrator.agents.items():
            for thought, act, obs in agent.scratchpad:
                logs.append(f"[{name}] Thought: {thought} | Action: {act} -> Obs: {obs}")
                
        return {"result": result, "logs": logs}
        
    # Ejecutamos de forma sincrona para la API REST
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    res = loop.run_until_complete(run_agents())
    loop.close()
    return res

# 14. Agentic Memory Layer (Ebbinghaus Decay)
@app.post("/api/memory/save")
def api_memory_save(req: MemorySaveRequest):
    if not AgenticMemory:
        return {"error": "AgenticMemory no importado."}
    memory = AgenticMemory(decay_factor=0.05, vector_dim=16)
    memory.save_fact(req.fact, importance=req.importance)
    # Extraemos hechos locales del fallback
    return {"status": "success", "saved_fact": req.fact}

@app.post("/api/memory/recall")
def api_memory_recall(req: MemoryRecallRequest):
    if not AgenticMemory:
        return {"error": "AgenticMemory no importado."}
    memory = AgenticMemory(decay_factor=req.decay_factor, vector_dim=16)
    
    # Poblamos de forma controlada modificando los timestamps de creacion para simular el olvido
    t0 = time.time() - req.time_elapsed
    memory.save_fact("Al usuario le gusta programar en Python", importance=8)
    memory.fallback_db[-1]["created_at"] = t0  # Retrocedemos en el tiempo
    
    results = memory.recall(req.query, top_k=1, current_time=time.time())
    return {"results": results}

# 15. Secure Tool Runtime Sandbox
@app.post("/api/runtime/run")
def api_runtime_run(req: RunCodeRequest):
    if not SecureToolRuntime:
        return {"error": "SecureToolRuntime no importado."}
    runtime = SecureToolRuntime()
    success, stdout, stderr, err = runtime.run_code(req.code, timeout_seconds=1.5)
    return {
        "success": success,
        "stdout": stdout,
        "stderr": stderr,
        "error": err
    }

# 16. Synthetic Data Generator
@app.post("/api/synthetic/generate")
def api_synthetic_generate(req: SyntheticRequest):
    if not SyntheticDataGenerator:
        return {"error": "SyntheticDataGenerator no importado."}
    generator = SyntheticDataGenerator(min_length=10, dedup_threshold=0.50)
    instructions = generator.generate_instruction_dataset(req.topics, count_per_topic=req.count)
    dpo_samples = generator.generate_dpo_dataset(req.topics, count_per_topic=req.count)
    
    return {
        "instructions": [{"prompt": i.instruction, "output": i.output} for i in instructions],
        "dpo": [{"prompt": d.prompt, "chosen": d.chosen, "rejected": d.rejected} for d in dpo_samples]
    }

# 17. Dataset Version Control (DVC)
@app.post("/api/dvc/commit")
def api_dvc_commit(req: DVCCommitRequest):
    if not DatasetVCS:
        return {"error": "DatasetVCS no importado."}
    vcs = DatasetVCS(store_dir=".dvc_hub_store")
    vcs.init()
    
    # Guardamos temporalmente el JSON
    filepath = req.filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(req.data, f)
        
    commit_hash = vcs.commit(filepath, req.message)
    return {"commit_hash": commit_hash, "log": vcs.log()}

@app.post("/api/dvc/diff")
def api_dvc_diff(req: Dict[str, str]):
    if not DatasetVCS:
        return {"error": "DatasetVCS no importado."}
    vcs = DatasetVCS(store_dir=".dvc_hub_store")
    res = vcs.diff(req["commit_a"], req["commit_b"])
    return res

@app.get("/api/dvc/log")
def api_dvc_log():
    if not DatasetVCS:
        return {"error": "DatasetVCS no importado."}
    vcs = DatasetVCS(store_dir=".dvc_hub_store")
    return vcs.log()

# 18. QLora Finetuner NF4 Simulator
@app.post("/api/qlora/train")
def api_qlora_train(req: QLoraTrainRequest):
    # Simula el bucle de entrenamiento y descenso de perdida
    loss_history = []
    current_loss = 0.85
    for epoch in range(1, req.epochs + 1):
        # Descenso de pérdida simulado
        current_loss -= float(np.random.uniform(0.1, 0.25))
        current_loss = max(0.005, current_loss)
        loss_history.append({"epoch": epoch, "loss": current_loss})
        
    return {
        "status": "success",
        "rank": req.r,
        "alpha": req.alpha,
        "loss_history": loss_history,
        "saved_adapters": {
            "lora_A": f"Weights shape: [{req.r}, 128]",
            "lora_B": f"Weights shape: [128, {req.r}]"
        }
    }

# 19. LLM Eval Harness
@app.post("/api/eval/benchmark")
def api_eval_benchmark(req: BenchmarkRequest):
    if not LLMEvaluator:
        return {"error": "LLMEvaluator no importado."}
    cand_tok = LLMEvaluator.clean_and_tokenize(req.candidate)
    ref_tok = LLMEvaluator.clean_and_tokenize(req.reference)
    
    exact = 1.0 if req.candidate.strip() == req.reference.strip() else 0.0
    jaccard = LLMEvaluator.compute_jaccard(cand_tok, ref_tok)
    cosine = LLMEvaluator.compute_cosine_similarity(cand_tok, ref_tok)
    bleu = LLMEvaluator.compute_bleu(cand_tok, ref_tok)
    
    ratio = len(cand_tok) / len(ref_tok) if ref_tok else 0.0
    
    return {
        "exact_match": exact,
        "jaccard_similarity": jaccard,
        "cosine_similarity": cosine,
        "bleu_score": bleu,
        "length_ratio": ratio
    }

# 20. Telemetry Trace
@app.get("/api/telemetry")
def api_telemetry():
    return global_tracer.get_trace_tree()

# 21. UNIFIED PIPELINE SIMULATOR (Concordancia completa de todos los módulos)
@app.post("/api/pipeline/run")
async def api_pipeline_run(req: UnifiedPipelineRequest):
    global_tracer.clear()
    
    # 1. Iniciar traza global
    global_tracer.start_span("AI Core Infra - Unified Pipeline Run", inputs={"prompt": req.prompt})
    
    # Paso 1: Interceptor Guardrails de Entrada
    global_tracer.start_span("1. Guardrails Input Scan (llm-guardrails-shield)")
    await asyncio.sleep(0.04)
    if LLMGuardrailsShield:
        shield = LLMGuardrailsShield()
        safe, clean, reason = shield.validate_input(req.prompt)
    else:
        safe, clean, reason = True, req.prompt, ""
    global_tracer.end_span(outputs={"is_safe": safe, "cleaned_prompt": clean})
    
    if not safe:
        global_tracer.end_span(error=Exception(f"Prompt bloqueado: {reason}"))
        return {"status": "blocked", "reason": reason, "telemetry": global_tracer.get_trace_tree()}
        
    # Paso 2: Router Semántico
    global_tracer.start_span("2. Complex Model Routing (semantic-model-router)")
    await asyncio.sleep(0.03)
    if SemanticModelRouter:
        router = SemanticModelRouter()
        decision = router.route(clean)
        model = decision.selected_model
        cost = decision.estimated_cost_per_1k_tokens
    else:
        model = "gpt-4o"
        cost = 0.005
    global_tracer.end_span(outputs={"selected_model": model, "estimated_cost": cost}, metadata_update={"cost": cost})
    
    # Paso 3: Embedding Generator
    global_tracer.start_span("3. Text Embedding Encoder (contrastive-embedding-trainer)")
    await asyncio.sleep(0.05)
    global_tracer.end_span(outputs={"vector_dimensions": 64})
    
    # Paso 4: Búsqueda Híbrida
    global_tracer.start_span("4. Hybrid Retrieval Search (hybrid-search-retrieval-pipeline)")
    await asyncio.sleep(0.08)
    # Simulamos recuperar chunks
    candidates = [
        {"id": "c1", "text": "La arquitectura RAG unifica busquedas lexicas y vectoriales en HNSW.", "score": 0.89},
        {"id": "c2", "text": "Los adaptadores LoRA e inferencia NF4 reducen los recursos de computo.", "score": 0.72}
    ]
    global_tracer.end_span(outputs={"candidates_retrieved": len(candidates)})
    
    # Paso 5: Reordenamiento Cross-Encoder
    global_tracer.start_span("5. Cross-Encoder Reranking (cross-encoder-reranker)")
    await asyncio.sleep(0.06)
    reranked = [candidates[0]] # Quedamos con el mejor
    global_tracer.end_span(outputs={"best_candidate": reranked[0]["id"]})
    
    # Paso 6: Memory Recall
    global_tracer.start_span("6. Episodic Memory Recall (agentic-memory-layer)")
    await asyncio.sleep(0.04)
    # Recuperar recuerdos temporales
    recalled_facts = ["Al usuario le interesa programar en Python"]
    global_tracer.end_span(outputs={"recalled_facts": recalled_facts})
    
    # Paso 7: Orchestra Agents ReAct Execution
    global_tracer.start_span("7. Agent ReAct Reasoning (orchestra-agents)")
    await asyncio.sleep(0.12)
    # Ejecucion simulada en sandbox
    global_tracer.start_span("7.1. Sandbox Safe Execution (secure-tool-runtime)")
    await asyncio.sleep(0.05)
    global_tracer.end_span(outputs={"stdout": "Resultado: 42", "success": True})
    
    agent_output = (
        "Basado en tus notas sobre RAG y NanoVectorDB, el pipeline híbrido recupera "
        "y fusiona de forma óptima los rankings. Además, de acuerdo a tu memoria, "
        "sé que prefieres implementarlo usando tipado estricto en Python. "
        "El código en el Sandbox se ejecutó con éxito."
    )
    global_tracer.end_span(outputs={"agent_response": agent_output})
    
    # Paso 8: Guardrail de Salida
    global_tracer.start_span("8. Guardrails Output Scan (llm-guardrails-shield)")
    await asyncio.sleep(0.04)
    if LLMGuardrailsShield:
        shield = LLMGuardrailsShield()
        context_str = " ".join([c["text"] for c in candidates])
        out_safe, clean_out, out_reason = shield.validate_output(agent_output, context_str)
    else:
        out_safe, clean_out, out_reason = True, agent_output, ""
    global_tracer.end_span(outputs={"is_safe": out_safe, "clean_response": clean_out})
    
    # Cierre de Span Raíz
    global_tracer.end_span()
    
    return {
        "status": "success",
        "model": model,
        "cost": cost,
        "response": clean_out,
        "telemetry": global_tracer.get_trace_tree()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
