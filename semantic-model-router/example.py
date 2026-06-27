import time
from router import SemanticModelRouter


def run_demo() -> None:
    print("=" * 65)
    print("      Demostracion de Enrutador Semantico de Modelos      ")
    print("=" * 65)
    
    router = SemanticModelRouter()
    
    prompts = [
        "hola",
        "Escribe un script en Python para calcular el PageRank de un grafo dirigido usando numpy.",
        "Analiza paso a paso las implicaciones filosoficas del libre albedrio en la era de la IA.",
        "Genera una consulta SQL para actualizar la tabla de empleados seteando el salario + 10%.",
        "gracias, adios!",
        "Explica detalladamente como optimizar la red siamesa del contrastive-embedding-trainer."
    ]
    
    total_cost_routed = 0.0
    total_cost_sonnet_only = 0.0
    
    # Asumimos que cada prompt + respuesta promedio tiene 1500 tokens (1000 input, 500 output)
    # Por lo tanto, el costo de una llamada es: 1.0 * cost_in_1k + 0.5 * cost_out_1k
    tokens_input_k = 1.0
    tokens_output_k = 0.5
    
    sonnet_profile = router.model_map["claude-3-5-sonnet"]
    cost_sonnet_single = (sonnet_profile.cost_per_1k_input * tokens_input_k) + (sonnet_profile.cost_per_1k_output * tokens_output_k)
    
    print(f"Estrategia Base: Enviar todo a Claude 3.5 Sonnet (Costo por peticion: ${cost_sonnet_single:.5f})\n")
    
    for i, p in enumerate(prompts, 1):
        decision = router.route(p)
        
        # Calcular el costo real de esta peticion en base al modelo elegido
        profile = router.model_map[decision.selected_model]
        cost_routed = (profile.cost_per_1k_input * tokens_input_k) + (profile.cost_per_1k_output * tokens_output_k)
        
        total_cost_routed += cost_routed
        total_cost_sonnet_only += cost_sonnet_single
        
        print(f"Peticion {i}: '{p[:65]}...'")
        print(f"  -> Categoria de Complejidad: {decision.complexity_category.upper()}")
        print(f"  -> Modelo Asignado: {decision.selected_model}")
        print(f"  -> Ruta de Decision: {decision.routing_path}")
        print(f"  -> Costo de esta peticion: ${cost_routed:.5f}")
        print(f"  -> Justificacion: {decision.reason}\n")
        
    savings = total_cost_sonnet_only - total_cost_routed
    savings_pct = (savings / total_cost_sonnet_only) * 100.0 if total_cost_sonnet_only > 0 else 0
    
    print("-" * 65)
    print(f"Costo Total usando Enrutador Semantico:  ${total_cost_routed:.5f}")
    print(f"Costo Total enviando solo a Sonnet:      ${total_cost_sonnet_only:.5f}")
    print(f"Ahorro Economico Estimado:                ${savings:.5f} ({savings_pct:.2f}% de ahorro)")
    print("-" * 65)


if __name__ == "__main__":
    run_demo()
