import os
import time
import random
from tracer import tracer, trace_span

# Semilla fija para consistencia en demostración
random.seed(42)

@trace_span("1. Escudo Guardrails (Entrada)")
def guardrails_shield_check(prompt: str) -> str:
    print("  [Shield] Validando inyecciones y datos sensibles...")
    time.sleep(0.08) # Simulación de latencia
    if "sk-" in prompt:
        raise ValueError("Fuga de API Key detectada en el prompt")
    return prompt

@trace_span("2. Enrutador Semántico")
def route_prompt(prompt: str) -> str:
    print("  [Router] Clasificando complejidad del prompt...")
    time.sleep(0.04)
    if len(prompt) > 50:
        return "gpt-4o-heavy"
    return "gpt-4o-mini"

@trace_span("3. Búsqueda Vectorial Híbrida")
def retrieve_documents(query: str) -> list:
    print("  [Retrieval] Buscando trozos en NanoVectorDB y BM25...")
    time.sleep(0.15)
    return ["Documento Chunk 1: Como entrenar LoRA...", "Documento Chunk 2: MLOps con DVC...", "Documento Chunk 3: Recetas de cocina..."]

@trace_span("4. Reranker (Cross-Encoder)")
def rerank_documents(query: str, documents: list) -> list:
    print("  [Rerank] Ordenando relevancia semántica de chunks...")
    time.sleep(0.12)
    # Devolvemos solo los 2 mejores ordenados
    return [documents[0], documents[1]]

def llm_generation_step(model: str, prompt: str, context: list) -> str:
    # Usamos start_span manual para demostrar la inyección de metadatos de tokens y costes
    tracer.start_span(
        name=f"5. Generación LLM ({model})",
        inputs={"model": model, "prompt": prompt, "context_length": len(context)}
    )
    print(f"  [LLM] Generando respuesta usando modelo '{model}'...")
    time.sleep(0.45) # Simulación de tiempo de generación
    
    response = "Para entrenar LoRA se congelan los pesos base y se optimizan las matrices A y B..."
    
    # Calcular tokens y costes aproximados
    input_tokens = len(prompt.split()) + len(str(context).split()) + 50
    output_tokens = len(response.split()) + 20
    
    # Precios simulados por token
    price_per_token_in = 0.000005 if "heavy" in model else 0.0000005
    price_per_token_out = 0.000015 if "heavy" in model else 0.0000015
    cost = (input_tokens * price_per_token_in) + (output_tokens * price_per_token_out)
    
    tracer.end_span(
        outputs={"response": response},
        metadata_update={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost
        }
    )
    return response

@trace_span("6. Escudo Guardrails (Salida)")
def guardrails_output_check(response: str) -> str:
    print("  [Shield] Comprobando toxicidad y alucinaciones en la respuesta...")
    time.sleep(0.06)
    return response

@trace_span("Flujo RAG Completo")
def execute_rag_pipeline(prompt: str) -> str:
    print(f"\nIniciando Pipeline de RAG para prompt: '{prompt}'")
    
    # 1. Entrada de seguridad
    clean_prompt = guardrails_shield_check(prompt)
    
    # 2. Enrutador
    model_choice = route_prompt(clean_prompt)
    
    # 3. Recuperación
    raw_docs = retrieve_documents(clean_prompt)
    
    # 4. Reranking
    filtered_docs = rerank_documents(clean_prompt, raw_docs)
    
    # 5. Generación
    response = llm_generation_step(model_choice, clean_prompt, filtered_docs)
    
    # 6. Salida de seguridad
    final_output = guardrails_output_check(response)
    
    print("Pipeline de RAG finalizado exitosamente.\n")
    return final_output


def run_demo() -> None:
    print("=" * 80)
    print("     Demostración de Observabilidad y Trazado de Spans en Pipelines de IA    ")
    print("=" * 80)
    
    # Limpiamos trazas previas
    tracer.clear()
    
    # Ejecutamos el flujo completo bajo un span raíz
    prompt_usuario = "Por favor explicame como entrenar LoRA de forma eficiente en PyTorch"
    execute_rag_pipeline(prompt_usuario)
    
    # 1. Mostrar Flame Graph en formato ASCII en la consola
    print("--- CRONOGRAMA DE EJECUCIÓN (Flame Graph ASCII) ---")
    ascii_flame = tracer.generate_ascii_flamegraph()
    print(ascii_flame)
    print("-" * 50)
    
    # 2. Exportar Flame Graph interactivo en formato HTML
    html_output_path = os.path.abspath("./observability_flamegraph.html")
    tracer.generate_html_flamegraph(html_output_path)
    
    print(f"\n[EXITO] Se ha exportado el Flame Graph interactivo de observabilidad.")
    print(f"Abra el archivo en su navegador para ver la linea temporal y detalles:")
    print(f"  file://{html_output_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
