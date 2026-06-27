import os
import sys
import uvicorn
from fastapi.testclient import TestClient

# Añadir path para importación del servidor
sys.path.append(os.path.dirname(__file__))

def bootstrap_data() -> None:
    """Pre-popula el cerebro digital con notas iniciales para construir un grafo rico."""
    from main import app
    client = TestClient(app)
    
    print("\n--- PASO 1: Pre-populando cerebro digital con notas iniciales ---")
    
    notes = [
        {
            "title": "Arquitectura RAG Híbrida",
            "content": "Para mejorar la precisión semántica y resolver términos específicos, nuestro pipeline de RAG combina búsquedas densas vectoriales en NanoVectorDB con búsquedas léxicas dispersas (BM25). Esto provee un contexto sumamente rico al generador."
        },
        {
            "title": "Ajuste Fino LoRA y QLoRA",
            "content": "LoRA es una técnica de PEFT para adaptar LLMs eficientemente. Congelamos la red y entrenamos matrices A y B. Con QLoRA, además cuantizamos los pesos a NF4 bloque-a-bloque para entrenar en GPUs estándar o simular localmente en PyTorch."
        },
        {
            "title": "Control de Versiones de Datos DVC",
            "content": "Para asegurar la reproducibilidad MLOps de entrenamiento, nuestro módulo DVC calcula firmas MD5 y guarda snapshots físicos indexados por commits. Esto nos permite volver atrás en el historial y calcular diferencias exactas de registros."
        }
    ]
    
    for note in notes:
        res = client.post("/api/notes", json=note)
        if res.status_code == 200:
            print(f"  Indexada exitosamente nota: '{note['title']}'")
        else:
            print(f"  Fallo al indexar nota '{note['title']}': {res.text}")
            
    print("Pre-populación finalizada. Grafo de conocimiento cargado.\n")


def start_server() -> None:
    """Lanza el servidor de producción local de Uvicorn."""
    print("=" * 80)
    # Explicación de la interconexión modular
    print("              NEXUS SECOND BRAIN - ECOSISTEMA INTEGRADO DE IA                ")
    print("=" * 80)
    print("Este producto consolida las siguientes arquitecturas desarrolladas en la infra:")
    print("  - Fragmentador Semántico (semantic-chunking-engine)")
    print("  - Indexador y Búsqueda Vectorial HNSW (nano-vector-db)")
    print("  - Reranker Cross-Encoder (cross-encoder-reranker)")
    print("  - Extracción de Entidades y Grafos (knowledge-graph-extractor)")
    print("  - Bucle de Agente Autónomo ReAct (orchestra-agents)")
    print("  - Almacén de Memoria a Corto y Largo Plazo (agentic-memory-layer)")
    print("  - Escudo Cortafuegos y Anonimizador PII (llm-guardrails-shield)")
    print("  - Trazado de Spans y Observabilidad (llm-observability-tracer)")
    print("-" * 80)
    
    # Pre-cargar datos
    bootstrap_data()
    
    print("Iniciando servidor web FastAPI local en:")
    print("  URL de Acceso: http://127.0.0.1:8000")
    print("  (Presione Ctrl+C para detener el servidor)")
    print("-" * 80)
    
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    start_server()
