

## Descripcion Tecnica y Alcance

El AI Core Infra Hub (ai-core-infra-hub) es una aplicacion de orquestacion y panel de control web unificado diseñado para integrar, ejecutar e inspeccionar las capacidades de los 21 subproyectos del repositorio ai-core-infra.

Este componente actua como la interfaz interactiva central que unifica el procesamiento de datos, el almacenamiento vectorial, el enrutamiento semantico de inferencias, los sistemas de agentes autonomos y las suites de calidad en una Single Page Application (SPA).

---

## Fundamento Matematico y Logica de Concordancia

El Hub coordina el flujo secuencial de datos de todos los modulos a traves de un simulador de pipeline completo, el cual calcula en tiempo real metricas de latencia y telemetria estructurada.

### 1. Curva de Retencion Episodica (Decaimiento de Ebbinghaus)
La memoria a largo plazo integra un factor de decaimiento temporal exponencial para determinar la recuperacion factica:

$$R(t) = e^{-\lambda \cdot \Delta t}$$

Donde $R(t)$ representa la retencion de memoria final, $\lambda$ es la tasa de olvido configurada y $\Delta t$ es el tiempo transcurrido (en segundos) desde la creacion del recuerdo.

### 2. Clasificacion de Trigramas y Distancia Jaccard
El enrutador semantico evalua la intencion de consultas rapidas calculando la similitud Jaccard de caracteres a nivel de tri-gramas:

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

Donde $A$ y $B$ son los conjuntos de tri-gramas extraidos de la consulta del usuario y del patron registrado.

---

## Estructuras de Datos y Configuracion

El Hub expone endpoints REST estructurados en FastAPI. A continuacion se detallan los esquemas clave:

### 1. Peticion del Pipeline Unificado (JSON Schema)
```json
{
  "type": "object",
  "properties": {
    "prompt": {
      "type": "string",
      "description": "Texto ingresado por el usuario"
    },
    "context": {
      "type": "string",
      "description": "Contexto factico adicional (opcional)"
    }
  },
  "required": ["prompt"]
}
```

### 2. Respuesta de la Inferencia (JSON Schema)
```json
{
  "type": "object",
  "properties": {
    "status": {
      "type": "string",
      "enum": ["success", "blocked"]
    },
    "model": {
      "type": "string"
    },
    "cost": {
      "type": "number"
    },
    "response": {
      "type": "string"
    },
    "telemetry": {
      "type": "array",
      "items": {
        "$ref": "#/definitions/SpanNode"
      }
    }
  }
}
```

---

## Instrucciones de Despliegue y Ejecucion

### Requisitos Previos
El Hub requiere Python 3.10+ y cuenta con su propio entorno virtual dedicado para asegurar aislamiento total de dependencias.

### Instalacion de Dependencias y Entorno Virtual
Para configurar el entorno virtual local e instalar las dependencias, ejecute en la carpeta del hub:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Ejecucion del Servidor
Arrancar la aplicacion mediante el script launcher:
```bash
python example.py
```
El panel estara disponible en la direccion `http://127.0.0.1:8000`.

### Ejecucion de Tests de Integracion
Para comprobar el correcto funcionamiento de las APIs REST:
```bash
python test_hub.py
```

---

## Interlinking y Arquitectura del Ecosistema (21 Proyectos Unificados)

El **AI Core Infra Hub** actúa como el gran director de orquesta e integrador del monorepo, consumiendo, importando u orquestando los **21 subproyectos** de la infraestructura de IA de forma integrada:

### 1. Ingesta y Procesamiento de Datos
*   **bpe-tokenizer-from-scratch:** Utilizado en `/api/bpe/tokenize` para pre-tokenizar textos en subpalabras y analizar estadísticas de codificación.
*   **semantic-chunking-engine:** Segmenta documentos dinámicamente en base a la distancia de similitud y desviación estándar en `/api/chunker/chunk`.
*   **multimodal-doc-parser:** Parser de documentos unificado en `/api/documents/upload-file` que extrae texto estructurado de PDFs complejos.
*   **llm-annotation-studio:** Simulación de estudio de etiquetado y corrección DPO en `/api/annotation/label` para optimización humana.

### 2. Almacenamiento, Representación y Recuperación (RAG)
*   **contrastive-embedding-trainer:** Provee el codificador de texto real (`SiameseTransformer`) de 768 dimensiones entrenado con Triplet Cosine Loss en PyTorch para generar embeddings vectoriales semánticos de producción.
*   **nano-vector-db:** Almacenamiento vectorial en memoria indexado por HNSW que indexa los READMEs de los subproyectos y los documentos cargados.
*   **knowledge-graph-extractor:** Extrae entidades y relaciones en formato de tripleta semántica en el endpoint `/api/knowledge-graph/extract`.
*   **hybrid-search-retrieval-pipeline:** Motor que realiza la búsqueda de dos fases combinando la base HNSW con el recuperador BM25 local.
*   **cross-encoder-reranker:** Re-ordena los candidatos recuperados en base a una puntuación de similitud semántica y léxica fina en `/api/pipeline/run`.

### 3. Inferencia, Ciclo de Vida y Seguridad
*   **llm-inference-server:** Servidor simulado de inferencia en `/api/inference/run` con capacidades de transmisión (streaming) de tokens.
*   **semantic-model-router:** Clasifica la complejidad semántica de los prompts de entrada y enruta la petición al perfil de modelo óptimo (coste vs latencia).
*   **llm-guardrails-shield:** Cortafuegos bidireccional en `/api/guardrails/check` que escanea prompts y respuestas buscando inyecciones de código, insultos o fugas de secretos (API keys).

### 4. Agentes Autónomos y Sandbox
*   **orchestra-agents:** Orquesta la ejecución final con bucles de pensamiento-acción del agente autónomo (ReAct) en el pipeline de RAG.
*   **agentic-memory-layer:** Conecta la memoria a largo plazo episódica utilizando decaimiento exponencial y la base de datos vectorial para dar persistencia al agente.
*   **secure-tool-runtime:** Simulación de entorno seguro y aislado (sandbox) en el que los agentes pueden ejecutar scripts de Python o APIs externas.

### 5. MLOps, Calidad e Historial
*   **synthetic-data-generator:** Motor de generación y filtrado de prompts sintéticos en `/api/synthetic/generate`.
*   **dataset-version-control:** Permite realizar comparativas (diffs) de registros estructurados y control de versiones en `/api/datasets/diff`.
*   **llm-qlora-finetuner:** Muestra curvas de pérdida histórica y simula el entrenamiento LoRA/PEFT en el endpoint `/api/finetuner/train`.
*   **llm-eval-harness:** Compara respuestas candidatas con referencias calculando métricas clásicas (BLEU, Jaccard, similitud coseno) en `/api/eval/benchmark`.
*   **llm-observability-tracer:** Motor de trazabilidad que mide tiempos, costes de tokens y latencias, renderizando el árbol jerárquico de Spans (telemetría unificada).

### 6. Interfaz e Integración Final
*   **nexus-second-brain:** Sirve de base conceptual para la interfaz SPA unificada (`index.html`) que integra los 21 módulos en un panel de control interactivo, con buscador en tiempo real, grafos visuales de telemetría y consola de chat del agente.
