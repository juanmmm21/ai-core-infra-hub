

## Technical Description and Scope

The AI Core Infra Hub (ai-core-infra-hub) is a unified orchestration and web dashboard application designed to integrate, execute, and inspect the capabilities of the 21 subprojects in the ai-core-infra repository.

This component acts as the central interactive interface that unifies data processing, vector storage, semantic inference routing, autonomous agent systems, and quality suites into a Single Page Application (SPA).

---

## Mathematical Foundation and Matching Logic

The Hub coordinates the sequential data flow of all modules through a full pipeline simulator, which computes latency metrics and structured telemetry in real time.

### 1. Episodic Retention Curve (Ebbinghaus Decay)
Long-term memory incorporates an exponential temporal decay factor to determine factual recall:

$$R(t) = e^{-\lambda \cdot \Delta t}$$

Where $R(t)$ represents the final memory retention, $\lambda$ is the configured forgetting rate, and $\Delta t$ is the elapsed time (in seconds) since the memory was created.

### 2. Trigram Classification and Jaccard Distance
The semantic router evaluates the intent of quick queries by computing the character-level Jaccard similarity at the trigram level:

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

Where $A$ and $B$ are the sets of trigrams extracted from the user's query and from the registered pattern.

---

## Data Structures and Configuration

The Hub exposes REST endpoints structured with FastAPI. The key schemas are detailed below:

### 1. Unified Pipeline Request (JSON Schema)
```json
{
  "type": "object",
  "properties": {
    "prompt": {
      "type": "string",
      "description": "Text entered by the user"
    },
    "context": {
      "type": "string",
      "description": "Additional factual context (optional)"
    }
  },
  "required": ["prompt"]
}
```

### 2. Inference Response (JSON Schema)
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

## Deployment and Run Instructions

### Prerequisites
The Hub requires Python 3.10+ and has its own dedicated virtual environment to ensure full dependency isolation.

### Installing Dependencies and the Virtual Environment
To set up the local virtual environment and install dependencies, run from the hub folder:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running the Server
Start the application via the launcher script:
```bash
python example.py
```
The dashboard will be available at `http://127.0.0.1:8000`.

### Running Integration Tests
To check that the REST APIs work correctly:
```bash
python test_hub.py
```

---

## Interlinking and Ecosystem Architecture (21 Unified Projects)

The **AI Core Infra Hub** acts as the grand conductor and integrator of the monorepo, consuming, importing, or orchestrating the AI infrastructure's **21 subprojects** in an integrated way:

### 1. Data Ingestion and Processing
*   **bpe-tokenizer-from-scratch:** Used at `/api/bpe/tokenize` to pre-tokenize text into subwords and analyze encoding statistics.
*   **semantic-chunking-engine:** Dynamically segments documents based on similarity distance and standard deviation at `/api/chunker/chunk`.
*   **multimodal-doc-parser:** Unified document parser at `/api/documents/upload-file` that extracts structured text from complex PDFs.
*   **llm-annotation-studio:** Labeling studio and DPO correction simulation at `/api/annotation/label` for human optimization.

### 2. Storage, Representation, and Retrieval (RAG)
*   **contrastive-embedding-trainer:** Provides the real text encoder (`SiameseTransformer`), a 768-dimensional model trained with Triplet Cosine Loss in PyTorch, to generate production-grade semantic vector embeddings.
*   **nano-vector-db:** In-memory vector storage indexed with HNSW that indexes the subprojects' READMEs and uploaded documents.
*   **knowledge-graph-extractor:** Extracts entities and relations in semantic triple format at the `/api/knowledge-graph/extract` endpoint.
*   **hybrid-search-retrieval-pipeline:** Engine that performs two-phase search, combining the HNSW base with the local BM25 retriever.
*   **cross-encoder-reranker:** Re-ranks retrieved candidates based on a fine-grained semantic and lexical similarity score at `/api/pipeline/run`.

### 3. Inference, Lifecycle, and Security
*   **llm-inference-server:** Simulated inference server at `/api/inference/run` with token streaming capability.
*   **semantic-model-router:** Classifies the semantic complexity of input prompts and routes the request to the optimal model profile (cost vs. latency).
*   **llm-guardrails-shield:** Bidirectional firewall at `/api/guardrails/check` that scans prompts and responses for code injections, insults, or secret leaks (API keys).

### 4. Autonomous Agents and Sandbox
*   **orchestra-agents:** Orchestrates final execution with the autonomous agent's thought-action loops (ReAct) in the RAG pipeline.
*   **agentic-memory-layer:** Connects episodic long-term memory using exponential decay and the vector database to give the agent persistence.
*   **secure-tool-runtime:** Simulation of a secure, isolated environment (sandbox) in which agents can execute Python scripts or external APIs.

### 5. MLOps, Quality, and History
*   **synthetic-data-generator:** Engine for generating and filtering synthetic prompts at `/api/synthetic/generate`.
*   **dataset-version-control:** Enables comparisons (diffs) of structured records and version control at `/api/datasets/diff`.
*   **llm-qlora-finetuner:** Shows historical loss curves and simulates LoRA/PEFT training at the `/api/finetuner/train` endpoint.
*   **llm-eval-harness:** Compares candidate responses against references, computing classic metrics (BLEU, Jaccard, cosine similarity) at `/api/eval/benchmark`.
*   **llm-observability-tracer:** Tracing engine that measures times, token costs, and latencies, rendering the hierarchical Span tree (unified telemetry).

### 6. Interface and Final Integration
*   **nexus-second-brain:** Serves as the conceptual foundation for the unified SPA interface (`index.html`) that integrates the 21 modules into an interactive dashboard, with real-time search, visual telemetry graphs, and an agent chat console.
