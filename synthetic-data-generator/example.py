import json
from generator import SyntheticDataGenerator


def run_demo() -> None:
    print("=" * 70)
    print("      Demostracion de Generacion de Datos Sinteticos (MLOps)      ")
    print("=" * 70)
    
    # 1. Inicializar el generador con filtros
    generator = SyntheticDataGenerator(min_length=15, dedup_threshold=0.45)
    
    topics = ["RAG", "embeddings", "db"]
    
    # 2. Generar Dataset de Instrucciones
    print("\nGenerando dataset sintético de Instrucciones (Instruction Tuning)...")
    instruction_data = generator.generate_instruction_dataset(topics, count_per_topic=4)
    
    instruction_list = [sample.model_dump() for sample in instruction_data]
    
    # Guardamos en archivo JSON local
    with open("synthetic_instruction_dataset.json", "w", encoding="utf-8") as f:
        json.dump(instruction_list, f, indent=2, ensure_ascii=False)
        
    print(f"  -> Dataset guardado exitosamente: 'synthetic_instruction_dataset.json'")
    print(f"  -> Total muestras generadas: {len(instruction_list)}")
    
    # Mostrar una muestra
    print(f"  -> Muestra 1:\n"
          f"     Instrucción: '{instruction_list[0]['instruction']}'\n"
          f"     Respuesta (Output): '{instruction_list[0]['output']}'\n")

    # 3. Generar Dataset DPO
    print("Generando dataset sintético para DPO (Direct Preference Optimization)...")
    dpo_data = generator.generate_dpo_dataset(topics, count_per_topic=4)
    
    dpo_list = [sample.model_dump() for sample in dpo_data]
    
    with open("synthetic_dpo_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dpo_list, f, indent=2, ensure_ascii=False)
        
    print(f"  -> Dataset DPO guardado exitosamente: 'synthetic_dpo_dataset.json'")
    print(f"  -> Total muestras generadas: {len(dpo_list)}")
    
    # Mostrar una muestra
    print(f"  -> Muestra DPO 1:\n"
          f"     Prompt: '{dpo_list[0]['prompt']}'\n"
          f"     Elegida (Chosen): '{dpo_list[0]['chosen']}'\n"
          f"     Rechazada (Rejected): '{dpo_list[0]['rejected']}'\n")


if __name__ == "__main__":
    run_demo()
