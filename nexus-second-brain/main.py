import os
import sys
import time
import math
import logging
from typing import Dict, List, Any, Tuple, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Configurar path para importar dependencias de proyectos hermanos (Interlinking)
sys.path.append(os.path.abspath("../llm-observability-tracer"))
sys.path.append(os.path.abspath("../nano-vector-db"))
sys.path.append(os.path.abspath("../semantic-chunking-engine"))
sys.path.append(os.path.abspath("../knowledge-graph-extractor"))
sys.path.append(os.path.abspath("../orchestra-agents"))

# Intentamos importar el trazador de observabilidad oficial
try:
    from tracer import tracer, trace_span
    logger_observability = True
except ImportError:
    # Fallback si no está disponible el proyecto hermano
    logger_observability = False
    def trace_span(name=None):
        def decorator(func):
            return func
        return decorator
    class MockTracer:
        def start_span(self, name, inputs=None, metadata=None): pass
        def end_span(self, outputs=None, error=None, metadata_update=None): pass
        def clear(self): pass
        def get_trace_tree(self): return []
    tracer = MockTracer()

# Inicialización de FastAPI
app = FastAPI(title="Nexus Second Brain - AI Core Infra SPA")

# Modelos de Datos para Peticiones
class NoteRequest(BaseModel):
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


# Base de datos en memoria para la demostración
notes_db: List[Dict[str, Any]] = []
knowledge_graph = {
    "nodes": [
        {"id": "Nexus", "group": 1, "type": "root", "val": 20},
        {"id": "IA Infrastructure", "group": 1, "type": "concept", "val": 15}
    ],
    "links": [
        {"source": "Nexus", "target": "IA Infrastructure", "value": 2}
    ]
}

# Variable global para guardar la última traza
last_trace: List[Dict[str, Any]] = []


# --- HELPER DE SIMULACIÓN DE EXTRACCIÓN DE ENTIDADES (Knowledge Graph) ---
def extract_entities_and_relations(title: str, content: str) -> List[Tuple[str, str, str]]:
    """
    Extractor heurístico simplificado para simular el parser de grafos de conocimiento.
    
    Analiza el texto buscando términos clave y genera relaciones lógicas para D3.js.
    """
    relations = []
    # Normalizar texto
    text = f"{title} {content}".lower()
    
    concepts = {
        "lora": ("LoRA", "PEFT", "optimiza"),
        "qlora": ("QLoRA", "Quantization", "comprime"),
        "pytorch": ("PyTorch", "Deep Learning", "desarrollado en"),
        "rag": ("RAG", "Search", "combina"),
        "vector": ("Vector DB", "Search", "almacena"),
        "embedding": ("Embeddings", "Vectores", "representa"),
        "guardrails": ("Guardrails", "Security", "protege"),
        "agent": ("Agents", "Autonomy", "ejecuta"),
        "MLOps": ("MLOps", "Workflow", "automatiza"),
        "dvc": ("DVC", "Datasets", "versiona")
    }
    
    # Agregar nodo del título de la nota
    for key, (node_name, category, relation_type) in concepts.items():
        if key in text:
            relations.append((title, relation_type, node_name))
            relations.append((node_name, "pertenece a", category))
            
    # Relación por defecto
    if not relations:
        relations.append((title, "contiene", "Información"))
        
    return relations


# --- ENDPOINTS ---

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Sirve la página web principal del SPA."""
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="Plantilla index.html no encontrada.")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/notes")
def get_notes():
    """Obtiene el listado completo de notas."""
    return notes_db


@app.post("/api/notes")
def create_note(request: NoteRequest):
    """
    Crea una nota, ejecuta fragmentación semántica y extrae entidades del grafo.
    
    Instrumentado con spans de observabilidad para monitorización en vivo.
    """
    global last_trace
    tracer.clear()
    
    # Iniciamos el span raíz para la ingesta de notas
    tracer.start_span("Ingesta y Procesamiento de Nota", inputs={"title": request.title})
    
    try:
        # Paso 1: Fragmentación Semántica
        tracer.start_span("1. Fragmentación Semántica (Semantic Chunking)")
        time.sleep(0.06) # Simulación de parsing y segmentación
        chunks = [
            f"Fragmento 1 de {request.title}: {request.content[:100]}...",
            f"Fragmento 2 de {request.title}: {request.content[100:200]}..."
        ] if len(request.content) > 100 else [request.content]
        tracer.end_span(outputs={"total_chunks": len(chunks)})
        
        # Paso 2: Generación de Embeddings e Indexación
        tracer.start_span("2. Indexación en Base de Datos Vectorial (NanoVectorDB)")
        time.sleep(0.08)
        tracer.end_span(outputs={"indexed_vectors": len(chunks)})
        
        # Paso 3: Extracción de Grafo de Conocimiento
        tracer.start_span("3. Extracción de Grafo (Knowledge Graph Extractor)")
        relations = extract_entities_and_relations(request.title, request.content)
        time.sleep(0.05)
        
        # Actualizar el grafo global
        existing_nodes = {node["id"] for node in knowledge_graph["nodes"]}
        
        # Insertar nodo de la nota
        if request.title not in existing_nodes:
            knowledge_graph["nodes"].append({"id": request.title, "group": 2, "type": "note", "val": 12})
            existing_nodes.add(request.title)
            
        for source, rel, target in relations:
            if source not in existing_nodes:
                knowledge_graph["nodes"].append({"id": source, "group": 3, "type": "concept", "val": 10})
                existing_nodes.add(source)
            if target not in existing_nodes:
                knowledge_graph["nodes"].append({"id": target, "group": 4, "type": "concept", "val": 10})
                existing_nodes.add(target)
                
            # Agregar link
            knowledge_graph["links"].append({"source": source, "target": target, "value": 1, "type": rel})
            
        tracer.end_span(outputs={"extracted_relations": len(relations)})
        
        # Guardamos la nota en la BD local
        note = {
            "id": len(notes_db) + 1,
            "title": request.title,
            "content": request.content,
            "chunks": chunks,
            "relations": relations,
            "timestamp": time.time()
        }
        notes_db.append(note)
        
        tracer.end_span(outputs={"status": "success", "note_id": note["id"]})
        last_trace = tracer.get_trace_tree()
        return {"status": "success", "note": note}
        
    except Exception as e:
        tracer.end_span(error=e)
        last_trace = tracer.get_trace_tree()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph")
def get_graph():
    """Devuelve los nodos y enlaces del grafo de conocimiento para D3.js."""
    return knowledge_graph


@app.get("/api/telemetry")
def get_telemetry():
    """Devuelve la última traza jerárquica para actualizar el panel de control."""
    return last_trace


@app.post("/api/chat")
async def chat_interaction(request: ChatRequest):
    """
    Endpoint del Agente Autónomo.
    
    Simula un ciclo de razonamiento ReAct instrumentado con trazas y
    retorna una respuesta en streaming usando Server-Sent Events (SSE).
    """
    global last_trace
    tracer.clear()
    
    # Iniciamos el span raíz para la consulta del agente
    tracer.start_span("Agente Autónomo ReAct - Consulta", inputs={"message": request.message})
    
    # Simulación de pensamientos y búsquedas
    tracer.start_span("1. Análisis e Intención del Agente")
    time.sleep(0.04)
    tracer.end_span(outputs={"intent": "búsqueda_seguridad_rag"})
    
    tracer.start_span("2. Escudo de Seguridad (Guardrails Shield)")
    time.sleep(0.05)
    # Comprobar inyecciones o keys
    if "sk-" in request.message:
        err = ValueError("API Key bloqueada en la consulta.")
        tracer.end_span(error=err)
        tracer.end_span(error=err)
        last_trace = tracer.get_trace_tree()
        raise HTTPException(status_code=400, detail="Inyección de API Key detectada y bloqueada.")
    tracer.end_span(outputs={"security_status": "CLEAN"})
    
    # Buscar en las notas locales (RAG)
    tracer.start_span("3. Búsqueda Híbrida en base de datos de Notas")
    time.sleep(0.08)
    # Encontramos notas similares por keywords simples
    matched_notes = []
    query_lower = request.message.lower()
    for note in notes_db:
        if note["title"].lower() in query_lower or any(word in note["content"].lower() for word in query_lower.split()):
            matched_notes.append(note["title"])
    tracer.end_span(outputs={"matched_notes": matched_notes, "count": len(matched_notes)})
    
    # Simulación de costes y tokens
    in_tokens = len(request.message.split()) + 40
    out_tokens = 80
    cost = (in_tokens * 0.000001) + (out_tokens * 0.000002)
    
    tracer.start_span(
        "4. Generación de Respuesta (LLM)",
        metadata={"input_tokens": in_tokens, "output_tokens": out_tokens, "cost": cost}
    )
    
    # Creamos un generador de streaming para simular SSE de tokens
    def sse_generator():
        # Primero enviamos una señal indicando que iniciamos el streaming
        yield "data: [START]\n\n"
        
        response_text = ""
        if matched_notes:
            response_text = f"Basándome en tus notas sobre '{', '.join(matched_notes)}', he encontrado la respuesta relevante a tu consulta. El grafo de conocimiento se ha actualizado reflejando la conexión semántica."
        else:
            response_text = "He analizado tu cerebro digital de notas pero no he encontrado datos relacionados. Sin embargo, en mi base de conocimientos general sé que este ecosistema de infraestructura de IA es altamente modular."
            
        words = response_text.split()
        for word in words:
            time.sleep(0.05) # Latencia de tokens simulados
            yield f"data: {word} \n\n"
            
        # Finalizar spans y guardar trazas
        tracer.end_span(outputs={"full_response": response_text})
        tracer.end_span(outputs={"status": "completed"})
        
        global last_trace
        last_trace = tracer.get_trace_tree()
        yield "data: [DONE]\n\n"
        
    return StreamingResponse(sse_generator(), media_type="text/event-stream")
