

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
El Hub requiere Python 3.10+ y acceso a las dependencias de los proyectos vecinos. Se recomienda utilizar el entorno virtual existente en `nexus-second-brain`.

### Instalacion de Dependencias del Hub
Para asegurar que todos los modulos adicionales esten disponibles, ejecute en la carpeta del hub:
```bash
../nexus-second-brain/.venv/bin/pip install -r requirements.txt
```

### Ejecucion del Servidor
Arrancar la aplicacion mediante el script launcher example:
```bash
../nexus-second-brain/.venv/bin/python example.py
```
El panel estara disponible en la direccion `http://127.0.0.1:8000`.

### Ejecucion de Tests de Integracion
Para comprobar el correcto enlazado y funcionamiento de las APIs REST:
```bash
../nexus-second-brain/.venv/bin/python test_hub.py
```

---

## Interlinking en el Ecosistema

El Hub importa dinamicamente las siguientes bibliotecas de la infraestructura:
*   **bpe-tokenizer-from-scratch** para analizar la division de tokens.
*   **semantic-chunking-engine** para realizar cortes de texto en base a desviacion estandar.
*   **nano-vector-db** para insertar y consultar embeddings vectoriales en memoria HNSW.
*   **hybrid-search-retrieval-pipeline** para realizar busquedas fusionadas con BM25 y RRF.
*   **llm-guardrails-shield** para filtrar prompts de entrada/salida y detectar PII o inyecciones.
*   **llm-observability-tracer** para capturar los Spans jerarquicos y renderizar el flame graph.
