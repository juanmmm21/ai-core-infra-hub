import os
import sys
import time
import math
import uuid
import json
import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import asyncio

# Setup path mapping to nested and sibling projects (Interlinking)
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
    nested_path = os.path.join(HUB_DIR, p)
    if os.path.isdir(nested_path):
        if nested_path not in sys.path:
            sys.path.append(nested_path)
    else:
        sibling_path = os.path.join(PARENT_DIR, p)
        if sibling_path not in sys.path:
            sys.path.append(sibling_path)


# Safe imports with placeholders and simulated fallbacks if necessary
import hashlib
import random
import re

try:
    from tokenizer import BPETokenizer
except ImportError:
    class BPETokenizer:
        def __init__(self):
            pass
        def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
            pass
        def encode(self, text: str) -> List[int]:
            return [ord(c) for c in text]
        def decode(self, tokens: List[int]) -> str:
            return "".join(chr(t) for t in tokens)

try:
    from chunker import SemanticChunker
    from embedding_provider import MockEmbeddingProvider
except ImportError:
    class MockEmbeddingProvider:
        def __init__(self, dimension: int = 384) -> None:
            self.dimension = dimension
        def get_embeddings(self, texts: List[str]) -> List[List[float]]:
            embeddings = []
            for text in texts:
                # Generate deterministic pseudo-embeddings based on text hash
                rng = random.Random(int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16))
                embeddings.append([rng.normalvariate(0.0, 1.0) for _ in range(self.dimension)])
            return embeddings

    class SemanticChunker:
        def __init__(self, embedding_provider, tokenizer=None, window_size=3, threshold_factor=1.2, max_tokens=512):
            self.embedding_provider = embedding_provider
        def chunk_text(self, text: str) -> List[Dict[str, Any]]:
            # Fallback basic chunking by sentences
            sentences = [s.strip() for s in re.split(r'(?<=\.|\?|!)\s+', text) if s.strip()]
            return [{"text": s, "sentences": [s], "index": i} for i, s in enumerate(sentences)]

try:
    from parser import MultimodalDocParser
    from backend import LocalExtractBackend
except ImportError:
    class LocalExtractBackend:
        pass
    class MultimodalDocParser:
        pass

try:
    from database import NanoVectorDB
except ImportError:
    class NanoVectorDB:
        def __init__(self, dimension: int = 64, index_type: str = "hnsw"):
            self.dimension = dimension
            self.index_type = index_type
            self.data = {}
        def insert(self, id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None):
            self.data[id] = {"vector": vector, "metadata": metadata or {}}
        def query(self, vector: List[float], top_k: int = 3, filter: Optional[Dict[str, Any]] = None):
            results = []
            v1 = np.array(vector)
            for item_id, item in self.data.items():
                v2 = np.array(item["vector"])
                # Cosine distance
                dot = np.dot(v1, v2)
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                sim = dot / (norm1 * norm2 + 1e-9)
                dist = 1.0 - float(sim)
                results.append({"id": item_id, "distance": dist, "metadata": item["metadata"]})
            results.sort(key=lambda x: x["distance"])
            return results[:top_k]

try:
    from extractor import KnowledgeGraphStore
except ImportError:
    class KnowledgeGraphStore:
        pass

try:
    from pipeline import HybridSearchPipeline
    from bm25 import BM25Retriever
except ImportError:
    class BM25Retriever:
        def __init__(self):
            self.corpus = {}
        def fit(self, corpus: Dict[str, str]):
            self.corpus = corpus
        def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[float, str]]:
            q_words = set(query.lower().split())
            results = []
            for doc_id, text in self.corpus.items():
                d_words = text.lower().split()
                score = sum(1.0 for w in q_words if w in d_words)
                results.append((score, doc_id))
            results.sort(reverse=True, key=lambda x: x[0])
            return results[:top_k]
    class HybridSearchPipeline:
        pass

try:
    from reranker import CrossEncoderReranker
except ImportError:
    class CrossEncoderReranker:
        def __init__(self, model_name: str, device: str = "cpu"):
            self.is_online = False
        def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
            q_words = set(query.lower().split())
            scored = []
            for doc in documents:
                d_words = doc["text"].lower().split()
                score = sum(0.3 for w in q_words if w in d_words) + 0.1
                doc_copy = doc.copy()
                doc_copy["rerank_score"] = min(0.99, score)
                scored.append(doc_copy)
            scored.sort(key=lambda x: x["rerank_score"], reverse=True)
            return scored[:top_k]

try:
    from inference_engine import InferenceEngine
except ImportError:
    class InferenceEngine:
        pass

try:
    from router import SemanticModelRouter, ModelProfile
except ImportError:
    class SemanticModelRouter:
        def route(self, prompt: str):
            length = len(prompt.split())
            if length < 6:
                class Decision:
                    selected_model = "gpt-4o-mini"
                    reason = "Consulta simple y corta (Simulado)"
                    estimated_cost_per_1k_tokens = 0.00015
                    complexity_category = "Baja"
                    routing_path = "Local rule-based heuristic -> gpt-4o-mini"
                return Decision()
            else:
                class Decision:
                    selected_model = "claude-3-5-sonnet"
                    reason = "Consulta compleja o requiere codigo (Simulado)"
                    estimated_cost_per_1k_tokens = 0.0030
                    complexity_category = "Alta"
                    routing_path = "Local rule-based heuristic -> claude-3-5-sonnet"
                return Decision()

try:
    from shield import LLMGuardrailsShield
except ImportError:
    class LLMGuardrailsShield:
        def validate_input(self, prompt: str) -> Tuple[bool, str, str]:
            # Redact basic phone numbers
            clean = prompt
            phone_pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,3}\)?[-.\s]?\d{3}[-.\s]?\d{3,4}\b"
            if re.search(phone_pattern, prompt):
                clean = re.sub(phone_pattern, "[PHONE]", prompt)
            
            # Simulated prompt injection check
            injection_patterns = [
                "ignore prior instructions", "delete database", "drop table",
                "elimina la base de datos", "elimina toda la base de datos",
                "borra la base de datos", "ignora las instrucciones"
            ]
            for pat in injection_patterns:
                if pat in prompt.lower():
                    return False, clean, f"Blocked: injection pattern '{pat}' detected (Simulado)"
            return True, clean, "Aprobado (Simulado)"
            
        def validate_output(self, response: str, context: Optional[str] = None) -> Tuple[bool, str, str]:
            if "sk-proj-" in response:
                return False, response.replace("sk-proj-", "sk-proj-[REDACTED]"), "Blocked: API key leak detected (Simulado)"
            return True, response, "Aprobado (Simulado)"

try:
    from agent import AutonomousAgent
    from orchestrator import AgentOrchestrator
except ImportError:
    class AutonomousAgent:
        def __init__(self, name: str, role: str, instruction: str, use_mock_llm: bool = True):
            self.name = name
            self.scratchpad = []
        def add_tool(self, name: str, func: Any):
            pass
    class AgentOrchestrator:
        def __init__(self, name: str):
            self.agents = {}
        def register_agent(self, agent: Any):
            self.agents[agent.name] = agent
        def execute(self, task: str) -> str:
            for name, agent in self.agents.items():
                agent.scratchpad.append((
                    f"Analizando como procesar '{task}' con rol de {agent.name}",
                    "search_db",
                    f"Resultados de busqueda mock del agente {name}"
                ))
            return f"Respuesta de orquestacion multi-agente simulada para: {task}."

try:
    from memory import AgenticMemory
except ImportError:
    class AgenticMemory:
        def __init__(self, decay_factor: float = 0.05, vector_dim: int = 16):
            self.decay_factor = decay_factor
            self.facts = []
        def save_fact(self, fact: str, importance: int = 5):
            self.facts.append({"fact": fact, "importance": importance, "created_at": time.time()})
        def recall(self, query: str, top_k: int = 1, current_time: Optional[float] = None) -> List[Dict[str, Any]]:
            if not current_time:
                current_time = time.time()
            results = []
            for item in self.facts:
                delta_t = current_time - item["created_at"]
                retention = math.exp(-self.decay_factor * delta_t)
                score = retention * (item["importance"] / 10.0)
                results.append({"fact": item["fact"], "score": score, "decayed_importance": item["importance"] * retention})
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

try:
    from runtime import SecureToolRuntime
except ImportError:
    class SecureToolRuntime:
        def run_code(self, code: str, timeout_seconds: float = 1.5) -> Tuple[bool, str, str, Optional[str]]:
            forbidden = ["os.", "sys.", "subprocess", "open("]
            for word in forbidden:
                if word in code:
                    return False, "", "", f"SecurityValidationError: Bloqueado por usar llamadas prohibidas '{word}' (Simulado)"
            return True, "Ejecucion en Sandbox exitosa (Simulado)\nOutput: 42", "", None

try:
    from generator import SyntheticDataGenerator
except ImportError:
    class SyntheticDataGenerator:
        def __init__(self, min_length: int = 10, dedup_threshold: float = 0.5):
            pass
        def generate_instruction_dataset(self, topics: List[str], count_per_topic: int = 2) -> List[Any]:
            class Item:
                def __init__(self, inst, out):
                    self.instruction = inst
                    self.output = out
            results = []
            for t in topics:
                for i in range(count_per_topic):
                    results.append(Item(f"Como funciona {t}?", f"Explicacion simulada {i} sobre {t}."))
            return results
        def generate_dpo_dataset(self, topics: List[str], count_per_topic: int = 2) -> List[Any]:
            class DpoItem:
                def __init__(self, p, c, r):
                    self.prompt = p
                    self.chosen = c
                    self.rejected = r
            results = []
            for t in topics:
                for i in range(count_per_topic):
                    results.append(DpoItem(f"Explicar {t}", f"Respuesta elegida y optimizada sobre {t}.", f"Respuesta rechazada o alucinada sobre {t}."))
            return results

try:
    from dvc import DatasetVCS
except ImportError:
    class DatasetVCS:
        def __init__(self, store_dir: str = ".dvc_hub_store"):
            self.commits = []
        def init(self):
            pass
        def commit(self, filename: str, message: str) -> str:
            commit_hash = hashlib.md5(f"{filename}-{message}-{time.time()}".encode()).hexdigest()[:8]
            self.commits.append({"hash": commit_hash, "message": message, "timestamp": time.time()})
            return commit_hash
        def diff(self, commit_a: str, commit_b: str) -> Dict[str, List[Any]]:
            return {"added": [], "removed": [], "modified": []}
        def log(self) -> List[Dict[str, Any]]:
            return self.commits

try:
    from finetuner import NF4Quantizer, LoraConfig
except ImportError:
    class NF4Quantizer:
        pass
    class LoraConfig:
        pass

try:
    from evaluator import LLMEvaluator
except ImportError:
    class LLMEvaluator:
        @staticmethod
        def clean_and_tokenize(text: str) -> List[str]:
            return text.lower().split()
        @staticmethod
        def compute_jaccard(tokens1: List[str], tokens2: List[str]) -> float:
            s1, s2 = set(tokens1), set(tokens2)
            return len(s1 & s2) / len(s1 | s2) if s1 | s2 else 0.0
        @staticmethod
        def compute_cosine_similarity(tokens1: List[str], tokens2: List[str]) -> float:
            s1, s2 = set(tokens1), set(tokens2)
            return len(s1 & s2) / (math.sqrt(len(s1)) * math.sqrt(len(s2)) + 1e-9)
        @staticmethod
        def compute_bleu(tokens1: List[str], tokens2: List[str]) -> float:
            s1, s2 = set(tokens1), set(tokens2)
            return len(s1 & s2) / len(s2) if s2 else 0.0

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

# Global pipeline orchestration state
global_db = None
global_bm25 = None
global_memory = None
global_shield = None
global_router = None
global_runtime = None
global_reranker = None
global_corpus = {}

# Fallback fusion functions if not imported
try:
    from fusion import reciprocal_rank_fusion, score_normalization_fusion
except ImportError:
    def reciprocal_rank_fusion(dense_ranks, sparse_ranks, k=60):
        rrf_scores = {}
        for rank, (score, doc_id) in enumerate(dense_ranks):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        for rank, (score, doc_id) in enumerate(sparse_ranks):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [(score, doc_id) for doc_id, score in sorted_docs]

    def score_normalization_fusion(dense_ranks, sparse_ranks, alpha=0.5, metric="cosine"):
        fused = {}
        def normalize(ranks):
            if not ranks:
                return {}
            scores = [s for s, d in ranks]
            min_s, max_s = min(scores), max(scores)
            rng = max_s - min_s
            if rng == 0:
                return {d: 1.0 for s, d in ranks}
            return {d: (s - min_s) / rng for s, d in ranks}
        dense_norm = normalize(dense_ranks)
        sparse_norm = normalize(sparse_ranks)
        all_docs = set(dense_norm.keys()) | set(sparse_norm.keys())
        for doc_id in all_docs:
            d_s = dense_norm.get(doc_id, 0.0)
            s_s = sparse_norm.get(doc_id, 0.0)
            fused[doc_id] = alpha * d_s + (1.0 - alpha) * s_s
        sorted_docs = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return [(score, doc_id) for doc_id, score in sorted_docs]

def init_global_pipeline():
    global global_db, global_bm25, global_memory, global_shield, global_router, global_runtime, global_corpus, global_reranker
    
    # 1. Initialize Vector DB HNSW and Chunker
    provider = MockEmbeddingProvider(dimension=64)
    global_db = NanoVectorDB(dimension=64, index_type="hnsw")
    chunker = SemanticChunker(embedding_provider=provider) if SemanticChunker else None
    
    # 2. Collect documentation files (our subprojects' READMEs) to build our corpus
    corpus = {}
    docs_to_index = []
    
    for p in PROJECT_PATHS:
        # Read nested readme
        readme_path = os.path.join(HUB_DIR, p, "README.md")
        if not os.path.exists(readme_path):
            readme_path = os.path.join(PARENT_DIR, p, "README.md")
            
        if os.path.exists(readme_path):
            try:
                with open(readme_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    
                if chunker:
                    chunks = chunker.chunk_text(text)
                    for i, c in enumerate(chunks):
                        doc_id = f"{p}_chunk_{i}"
                        corpus[doc_id] = c["text"]
                        docs_to_index.append((doc_id, c["text"], {"source": f"{p}/README.md"}))
                else:
                    sentences = text.split("\n\n")
                    for i, s in enumerate(sentences):
                        if s.strip():
                            doc_id = f"{p}_sentence_{i}"
                            corpus[doc_id] = s.strip()
                            docs_to_index.append((doc_id, s.strip(), {"source": f"{p}/README.md"}))
            except Exception as e:
                logger.error(f"Error indexing {p} README: {e}")
                
    # Also index PROJECTS.md
    projects_file = os.path.join(PARENT_DIR, "PROJECTS.md")
    if os.path.exists(projects_file):
        try:
            with open(projects_file, "r", encoding="utf-8") as f:
                text = f.read()
            if chunker:
                chunks = chunker.chunk_text(text)
                for i, c in enumerate(chunks):
                    doc_id = f"projects_chunk_{i}"
                    corpus[doc_id] = c["text"]
                    docs_to_index.append((doc_id, c["text"], {"source": "PROJECTS.md"}))
            else:
                sentences = text.split("\n\n")
                for i, s in enumerate(sentences):
                    if s.strip():
                        doc_id = f"projects_sentence_{i}"
                        corpus[doc_id] = s.strip()
                        docs_to_index.append((doc_id, s.strip(), {"source": "PROJECTS.md"}))
        except Exception as e:
            logger.error(f"Error indexing PROJECTS.md: {e}")
            
    # Index documents into HNSW NanoVectorDB
    for doc_id, text_content, meta in docs_to_index:
        vec = provider.get_embeddings([text_content])[0]
        vec_64 = vec[:64] if len(vec) >= 64 else vec + [0.0]*(64-len(vec))
        global_db.insert(id=doc_id, vector=vec_64, metadata=meta)
        
    global_corpus = corpus
    
    # 3. Fit BM25
    global_bm25 = BM25Retriever()
    if corpus:
        global_bm25.fit(corpus)
    else:
        global_bm25.fit({"doc_default": "La infraestructura unificada de IA conecta todos los submodulos."})
        
    # 4. Reranker
    global_reranker = CrossEncoderReranker(model_name="dummy")
    
    # 5. Episodic Memory
    global_memory = AgenticMemory(decay_factor=0.01, vector_dim=16)
    global_memory.save_fact("El usuario prefiere respuestas tecnicas y explicaciones con formulas matematicas.", importance=9)
    global_memory.save_fact("Los subproyectos de la infraestructura estan construidos artesanalmente en Python.", importance=8)
    
    # 5. Safety & Router
    global_shield = LLMGuardrailsShield()
    global_router = SemanticModelRouter()
    
    # 6. Secure Runtime
    global_runtime = SecureToolRuntime()
    
    logger.info(f"[+] Global pipeline initialized successfully. Corpus size: {len(corpus)} documents.")

@app.on_event("startup")
def startup_event():
    init_global_pipeline()

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
def generate_agent_response(prompt: str, context: str, memories: List[str]) -> str:
    prompt_clean = prompt.strip()
    
    # Conversational greetings check
    greetings = ["hola", "que tal", "buenos dias", "buenos días", "buenas tardes", "buenas noches", "hello", "hi"]
    is_greeting = any(g in prompt_clean.lower() for g in greetings)
    
    if is_greeting:
        return (
            "¡Hola! ¿Cómo estás? Soy el asistente virtual del AI Core Infra Hub.\n"
            "He procesado tu saludo y veo que es una consulta de baja complejidad. "
            "El enrutador semántico me ha asignado un modelo rápido y económico (como gpt-4o-mini) para responderte.\n"
            "¿En qué puedo ayudarte hoy en relación a la infraestructura de IA (tokenizadores, chunking, bases vectoriales, agentes o guardrails)?"
        )
        
    # Helper to normalize accents and casing for accurate overlap calculations
    def normalize_text(t: str) -> str:
        accents = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u", "ñ": "n"}
        res = t.lower()
        for a, b in accents.items():
            res = res.replace(a, b)
        return res
        
    # Analyze and parse the retrieved context documents
    import re
    docs = re.findall(r"Documento \[(.*?)\]:\n(.*?)(?=\n\nDocumento |$)", context, re.DOTALL)
    
    summary = f"Procesando consulta: '{prompt_clean}'\n\n"
    summary += "### 🔍 Resultados de Búsqueda Semántica RAG:\n"
    
    if docs:
        summary += "He recuperado y ordenado los siguientes fragmentos relevantes de la documentación del proyecto:\n\n"
        for doc_id, doc_text in docs[:2]: # Show top 2 matches
            clean_text = doc_text.strip()
            if len(clean_text) > 400:
                excerpt = clean_text[:400] + "..."
            else:
                excerpt = clean_text
            summary += f"📄 **[`{doc_id}`]**\n> {excerpt}\n\n"
            
        summary += "---\n"
        
        # 1. Extractive Heuristic QA
        # We split doc content into sentences and find those that contain query keywords
        prompt_normalized = normalize_text(prompt_clean)
        prompt_words = [w.strip("?,.:;!\"'()").lower() for w in prompt_normalized.split() if len(w) > 2]
        stopwords = {"que", "las", "los", "del", "con", "por", "para", "una", "uno", "unos", "unas", "este", "esta", "como", "pero", "donde", "cuando", "quien", "hay", "esta", "estan", "esta", "sobre", "hay", "mas"}
        query_keywords = [w for w in prompt_words if w not in stopwords]
        
        matched_sentences = []
        for doc_id, doc_text in docs:
            # Simple sentence tokenizer by period or question mark followed by whitespace
            sentences = re.split(r'(?<=[.?!])\s+', doc_text)
            for sent in sentences:
                sent_clean = sent.strip()
                if not sent_clean or len(sent_clean) < 10:
                    continue
                sent_norm = normalize_text(sent_clean)
                sent_words = [w.strip("?,.:;!\"'()").lower() for w in sent_norm.split()]
                overlap = sum(1 for kw in query_keywords if kw in sent_words)
                if overlap > 0:
                    matched_sentences.append((overlap, sent_clean, doc_id))
                    
        # Sort by overlap score descending, then sentence length descending
        matched_sentences.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
        
        # 2. Compile response
        summary += "### 💡 Respuesta y Recomendación del Agente:\n"
        if matched_sentences:
            summary += (
                "De acuerdo a los extractos de los documentos indexados, "
                "aquí están los pasajes exactos que responden a tu consulta:\n\n"
            )
            # Take top 3 matching sentences
            seen_sents = set()
            count = 0
            for score, sent, doc_id in matched_sentences:
                sent_norm = sent.lower().strip()
                if sent_norm in seen_sents:
                    continue
                seen_sents.add(sent_norm)
                summary += f"👉 *\"{sent}\"* (encontrado en `{doc_id}`)\n"
                count += 1
                if count >= 3:
                    break
                    
            summary += "\n---\n*Nota: Estos fragmentos fueron extraídos semánticamente en tiempo real de tu base de conocimiento local utilizando el segmentador y recuperador de la infraestructura.*"
        else:
            # Fallback to custom keywords check if no direct sentence matches
            lower_prompt = prompt_clean.lower()
            if any(w in lower_prompt for w in ["documento", "meter", "entrenar", "pdf", "procesar", "ingestar"]):
                summary += (
                    "Para alimentar nuevos documentos e indexarlos en la infraestructura de IA, debes seguir este flujo de trabajo:\n"
                    "1. **Extracción y Parseo:** Usa el módulo `multimodal-doc-parser` para convertir PDFs o imágenes a Markdown estructurado utilizando modelos VLM.\n"
                    "2. **Segmentación:** Fragmenta el texto con el módulo `semantic-chunking-engine` para obtener chunks basados en similitud semántica.\n"
                    "3. **Entrenamiento y Embeddings:** Pasa los chunks por el codificador de `contrastive-embedding-trainer` para generar sus vectores representativos.\n"
                    "4. **Almacenamiento e Indexación:** Inserta los vectores en la base de datos `nano-vector-db` configurada con índice HNSW para habilitar búsquedas ultra-rápidas."
                )
            elif any(w in lower_prompt for w in ["vector", "hnsw", "db", "almacenar", "base de datos"]):
                summary += (
                    "Para interactuar con la base de datos vectorial del proyecto:\n"
                    "1. Crea una base de datos con `db = NanoVectorDB(dimension=64, index_type='hnsw')`.\n"
                    "2. Agrega vectores llamando a `db.insert(id=doc_id, vector=vector_float, metadata=meta_dict)`.\n"
                    "3. Realiza búsquedas aproximadas de vecinos más cercanos (ANN) usando `db.query(vector=query_vector, top_k=3)`."
                )
            elif any(w in lower_prompt for w in ["chunk", "semantic", "segmentar", "fragmentar"]):
                summary += (
                    "El módulo `semantic-chunking-engine` segmenta textos analizando la distancia coseno entre oraciones adyacentes.\n"
                    "Configura una instancia con un umbral dinámico (media + threshold * desviación estándar) para agrupar oraciones en párrafos sin romper la coherencia semántica."
                )
            elif any(w in lower_prompt for w in ["guardrail", "seguridad", "proteg", "filtra"]):
                summary += (
                    "Para proteger tu aplicación:\n"
                    "1. Instancia el escudo con `shield = LLMGuardrailsShield()`.\n"
                    "2. Llama a `shield.validate_input(prompt)` para filtrar inyecciones, jailbreaks y anonimizar teléfonos/emails.\n"
                    "3. Valida las salidas generadas mediante `shield.validate_output(response, context)` para evitar alucinaciones."
                )
            else:
                summary += (
                    "He recuperado la información relevante sobre este tema del repositorio. "
                    "Puedes consultar el detalle completo de la implementación revisando los README correspondientes a los módulos citados arriba."
                )
    else:
        summary += (
            "No se han encontrado fragmentos específicos en PROJECTS.md relacionados directamente con tu consulta.\n\n"
            "Por favor, intenta preguntar acerca de conceptos específicos del proyecto como 'NanoVectorDB', 'Semantic Chunker', 'Guardrails', 'ReAct Agents' o 'Embeddings'."
        )
        
    return summary


@app.post("/api/documents/upload-file")
async def api_documents_upload_file(file: UploadFile = File(...)):
    global global_db, global_bm25, global_corpus
    if global_db is None or global_bm25 is None:
        init_global_pipeline()
        
    filename = file.filename
    _, ext = os.path.splitext(filename.lower())
    
    content_text = ""
    try:
        if ext == ".pdf":
            import fitz
            pdf_bytes = await file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages_text = []
            for page in doc:
                pages_text.append(page.get_text())
            doc.close()
            content_text = "\n\n".join(pages_text)
        elif ext in [".txt", ".md", ".json", ".py", ".js", ".csv", ".yaml", ".yml", ".ini"]:
            file_bytes = await file.read()
            content_text = file_bytes.decode("utf-8", errors="ignore")
        else:
            return {"error": f"Formato de archivo '{ext}' no soportado. Sube un PDF, TXT, MD, etc."}
    except Exception as e:
        return {"error": f"Error al leer el archivo: {str(e)}"}
        
    if not content_text.strip():
        return {"error": "El archivo está vacío o no contiene texto extraíble."}
        
    provider = MockEmbeddingProvider(dimension=64)
    chunker = SemanticChunker(embedding_provider=provider) if SemanticChunker else None
    
    chunks_list = []
    if chunker:
        try:
            chunks = chunker.chunk_text(content_text)
            chunks_list = [c["text"] for c in chunks]
        except Exception as e:
            logger.error(f"Error chunking document file: {e}")
            chunks_list = [content_text]
    else:
        chunks_list = [s.strip() for s in content_text.split("\n\n") if s.strip()]
        
    if not chunks_list:
        chunks_list = [content_text]
        
    title_clean = filename.replace(" ", "_").lower()
    for i, chunk_text in enumerate(chunks_list):
        doc_id = f"file_{title_clean}_chunk_{i}"
        vec = provider.get_embeddings([chunk_text])[0]
        vec_64 = vec[:64] if len(vec) >= 64 else vec + [0.0]*(64-len(vec))
        global_db.insert(id=doc_id, vector=vec_64, metadata={"source": filename})
        global_corpus[doc_id] = chunk_text
        
    global_bm25.fit(global_corpus)
    
    return {
        "status": "success",
        "filename": filename,
        "chunks_count": len(chunks_list),
        "total_chars": len(content_text)
    }

@app.post("/api/documents/upload")
def api_documents_upload(req: DocumentUploadRequest):
    global global_db, global_bm25, global_corpus
    if global_db is None or global_bm25 is None:
        init_global_pipeline()
        
    title_clean = req.title.strip().replace(" ", "_").lower()
    content_text = req.content.strip()
    
    # 1. Run Chunker
    provider = MockEmbeddingProvider(dimension=64)
    chunker = SemanticChunker(embedding_provider=provider) if SemanticChunker else None
    
    chunks_list = []
    if chunker:
        try:
            chunks = chunker.chunk_text(content_text)
            chunks_list = [c["text"] for c in chunks]
        except Exception as e:
            logger.error(f"Error chunking uploaded document: {e}")
            chunks_list = [content_text]
    else:
        chunks_list = [s.strip() for s in content_text.split("\n\n") if s.strip()]
        
    if not chunks_list:
        chunks_list = [content_text]
        
    # 2. Insert into database (HNSW) and global_corpus (BM25)
    for i, chunk_text in enumerate(chunks_list):
        doc_id = f"uploaded_{title_clean}_chunk_{i}"
        
        # Calculate embedding vector
        vec = provider.get_embeddings([chunk_text])[0]
        vec_64 = vec[:64] if len(vec) >= 64 else vec + [0.0]*(64-len(vec))
        
        # Insert
        global_db.insert(id=doc_id, vector=vec_64, metadata={"source": req.title})
        global_corpus[doc_id] = chunk_text
        
    # 3. Re-fit BM25
    global_bm25.fit(global_corpus)
    
    return {"status": "success", "chunks_count": len(chunks_list)}

@app.get("/api/vector-db/list")
def api_vector_db_list():
    global global_db, global_corpus
    if global_db is None:
        init_global_pipeline()
        
    records = []
    # If using real NanoVectorDB (with self.vectors and self.metadata)
    if hasattr(global_db, "vectors") and hasattr(global_db, "metadata"):
        for doc_id, vec in global_db.vectors.items():
            meta = global_db.metadata.get(doc_id, {})
            text = global_corpus.get(doc_id, "Texto no indexado en caché local")
            # Convert numpy array to list if needed
            vec_list = vec.tolist() if hasattr(vec, "tolist") else list(vec)
            records.append({
                "id": str(doc_id),
                "metadata": meta,
                "text": text,
                "vector_preview": vec_list[:5]
            })
    # If using fallback NanoVectorDB (with self.data)
    elif hasattr(global_db, "data"):
        for doc_id, item in global_db.data.items():
            meta = item.get("metadata", {})
            vec = item.get("vector", [])
            text = global_corpus.get(doc_id, "Texto no indexado en caché local")
            records.append({
                "id": str(doc_id),
                "metadata": meta,
                "text": text,
                "vector_preview": list(vec[:5])
            })
            
    return {"status": "success", "records": records}

@app.post("/api/pipeline/run")
async def api_pipeline_run(req: UnifiedPipelineRequest):
    global global_shield, global_db, global_bm25, global_memory, global_router, global_runtime, global_reranker, global_corpus
    if global_shield is None:
        init_global_pipeline()
    global_tracer.clear()
    
    # 1. Iniciar traza global
    global_tracer.start_span("AI Core Infra - Unified Pipeline Run", inputs={"prompt": req.prompt})
    
    # Paso 1: Interceptor Guardrails de Entrada (Shield real)
    global_tracer.start_span("1. Guardrails Input Scan (llm-guardrails-shield)")
    safe, clean, reason = global_shield.validate_input(req.prompt)
    global_tracer.end_span(outputs={"is_safe": safe, "cleaned_prompt": clean})
    
    if not safe:
        global_tracer.end_span(error=Exception(f"Prompt bloqueado: {reason}"))
        return {"status": "blocked", "reason": reason, "telemetry": global_tracer.get_trace_tree()}
        
    # Paso 2: Router Semántico (Router real)
    global_tracer.start_span("2. Complex Model Routing (semantic-model-router)")
    decision = global_router.route(clean)
    model = decision.selected_model
    cost = decision.estimated_cost_per_1k_tokens
    global_tracer.end_span(outputs={"selected_model": model, "complexity": decision.complexity_category})
    
    # Paso 3: Embedding Generator (Mock real de 64 dimensiones)
    global_tracer.start_span("3. Text Embedding Encoder (contrastive-embedding-trainer)")
    provider = MockEmbeddingProvider(dimension=64)
    q_vec = provider.get_embeddings([clean])[0]
    q_vec_64 = q_vec[:64] if len(q_vec) >= 64 else q_vec + [0.0]*(64-len(q_vec))
    global_tracer.end_span(outputs={"vector_dimensions": len(q_vec_64)})
    
    # Paso 4: Búsqueda Híbrida (Pipeline real sobre PROJECTS.md)
    global_tracer.start_span("4. Hybrid Retrieval Search (hybrid-search-retrieval-pipeline)")
    
    # 4.1. Sparse Retrieval (BM25 real)
    sparse_res = global_bm25.retrieve(clean, top_k=3)
    
    # 4.2. Dense Retrieval (HNSW NanoVectorDB real)
    dense_raw = global_db.query(vector=q_vec_64, top_k=3)
    dense_res = [(r["distance"], r["id"]) for r in dense_raw]
    
    # 4.3. Fusion (Score Fusion real)
    fused_results = score_normalization_fusion(dense_res, sparse_res, alpha=0.5, metric="cosine")
    
    candidates = []
    for score, doc_id in fused_results:
        content = global_corpus.get(doc_id, "Documento de la infraestructura de IA")
        candidates.append({"id": doc_id, "text": content, "score": score})
        
    global_tracer.end_span(outputs={"candidates_retrieved": len(candidates)})
    
    # Paso 5: Reordenamiento Cross-Encoder (Reranker real)
    global_tracer.start_span("5. Cross-Encoder Reranking (cross-encoder-reranker)")
    reranked = global_reranker.rerank(clean, candidates, top_k=2)
    global_tracer.end_span(outputs={"reranked_candidates": [{"id": r["id"], "score": r["rerank_score"]} for r in reranked]})
    
    # Paso 6: Memory Recall (Episodic Memory real)
    global_tracer.start_span("6. Episodic Memory Recall (agentic-memory-layer)")
    recalled_facts = global_memory.recall(clean, top_k=2)
    global_tracer.end_span(outputs={"recalled_facts": [f["fact"] for f in recalled_facts]})
    
    # Paso 7: Orchestra Agents ReAct Execution (Agente real)
    global_tracer.start_span("7. Agent ReAct Reasoning (orchestra-agents)")
    
    # Unificamos el contexto recuperado
    context_blocks = []
    for r in reranked:
        context_blocks.append(f"Documento [{r['id']}]:\n{r['text']}")
    for m in recalled_facts:
        context_blocks.append(f"Recuerdo de Memoria:\n{m['fact']}")
    context_str = "\n\n".join(context_blocks)
    
    # 7.1. Sandbox Safe Execution (Secure tool runtime real)
    global_tracer.start_span("7.1. Sandbox Safe Execution (secure-tool-runtime)")
    test_code = "def check_env():\n    return 'Entorno Sandbox Seguro Verificado'\nprint(check_env())"
    success, stdout, stderr, err = global_runtime.run_code(test_code, timeout_seconds=1.5)
    global_tracer.end_span(outputs={"success": success, "stdout": stdout.strip()})
    
    # Generamos la respuesta integrada
    agent_output = generate_agent_response(clean, context_str, [f["fact"] for f in recalled_facts])
    global_tracer.end_span(outputs={"agent_response": agent_output})
    
    # Paso 8: Guardrail de Salida (Shield real)
    global_tracer.start_span("8. Guardrails Output Scan (llm-guardrails-shield)")
    out_safe, clean_out, out_reason = global_shield.validate_output(agent_output, context_str)
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
