import time
from memory import AgenticMemory


def run_demo() -> None:
    print("=" * 70)
    print("      Demostracion de Capa de Memoria de Agente (Ebbinghaus Decay)      ")
    print("=" * 70)
    
    # 1. Inicializar la memoria con decaimiento temporal modelado
    # decay_factor = 0.02 (decae moderadamente con el tiempo)
    memory = AgenticMemory(short_term_max_turns=3, decay_factor=0.02, vector_dim=32)
    
    # Simular marca de tiempo inicial (t0)
    t0 = time.time()
    
    print("\n--- PASO 1: Guardando hechos de largo plazo en t0 ---")
    memory.save_fact("El usuario trabaja como desarrollador backend de Python en Madrid", importance=9)
    memory.save_fact("Al usuario le gusta tomar cafe espresso doble por las mañanas", importance=6)
    memory.save_fact("El usuario planea viajar de vacaciones a Japon en Diciembre", importance=7)
    
    # 2. Recall inmediato en t = t0
    print("\n--- PASO 2: Recuperacion inmediata en t = t0 (Sin retraso) ---")
    query = "Habitos de cafe y bebidas del usuario"
    results_t0 = memory.recall(query, top_k=2, current_time=t0)
    
    for i, res in enumerate(results_t0, 1):
        print(f"Recuerdo {i}: '{res['fact']}'")
        print(f"  -> Importancia original: {res['importance']}")
        print(f"  -> Score de Recuerdo: {res['recall_score']:.4f}\n")
        
    # 3. Simular el paso del tiempo: Consulta despues de 30 segundos
    t30 = t0 + 30.0
    print(f"\n--- PASO 3: Recuperacion despues de 30 segundos (Decaimiento Temporal) ---")
    print(f"Simulando delta_t = 30 segundos...")
    results_t30 = memory.recall(query, top_k=2, current_time=t30)
    
    for i, res in enumerate(results_t30, 1):
        print(f"Recuerdo {i}: '{res['fact']}'")
        print(f"  -> Importancia original: {res['importance']}")
        print(f"  -> Score de Recuerdo (Decaido): {res['recall_score']:.4f}")
        # Comparativa con t0
        orig_score = next(r["recall_score"] for r in results_t0 if r["fact"] == res["fact"])
        pct_retention = (res["recall_score"] / orig_score) * 100.0
        print(f"  -> Retencion de memoria: {pct_retention:.2f}%\n")
        
    # 4. Simular el paso de mucho tiempo: Consulta despues de 10 minutos (600 segundos)
    t600 = t0 + 600.0
    print(f"\n--- PASO 4: Recuperacion despues de 10 minutos (Olvido extremo) ---")
    print(f"Simulando delta_t = 600 segundos...")
    results_t600 = memory.recall(query, top_k=2, current_time=t600)
    
    for i, res in enumerate(results_t600, 1):
        print(f"Recuerdo {i}: '{res['fact']}'")
        print(f"  -> Score de Recuerdo (Muy Bajo): {res['recall_score']:.4f}")
        orig_score = next(r["recall_score"] for r in results_t0 if r["fact"] == res["fact"])
        pct_retention = (res["recall_score"] / orig_score) * 100.0
        print(f"  -> Retencion de memoria: {pct_retention:.4f}%\n")


if __name__ == "__main__":
    run_demo()
