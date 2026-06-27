import json
import os
import shutil
from dvc import DatasetVCS


def run_demo() -> None:
    print("=" * 70)
    print("      Demostracion de Control de Versiones de Datos (VCS)      ")
    print("=" * 70)
    
    # 1. Definir rutas locales
    store_dir = ".dvc_store_example"
    dataset_file = "instruction_dataset.json"
    
    # Limpiamos ejecuciones previas de ejemplo
    for path in [store_dir, dataset_file]:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
                
    # 2. Inicializar VCS de datos
    vcs = DatasetVCS(store_dir=store_dir)
    vcs.init()
    
    # 3. Crear Dataset Version 1
    dataset_v1 = [
        {"instruction": "Escribe una funcion recursiva en Python", "output": "def f(n): ..."},
        {"instruction": "Calcula el embedding de RAG hibrido", "output": "vector = [...]"},
        {"instruction": "Como indexar grafos HNSW?", "output": "HNSW indexa en capas..."}
    ]
    
    with open(dataset_file, "w", encoding="utf-8") as f:
        json.dump(dataset_v1, f, indent=2, ensure_ascii=False)
        
    print("\n--- PASO 1: Creando y confirmando Version 1 del Dataset ---")
    commit_v1 = vcs.commit(dataset_file, "v1: Cargar 3 instrucciones iniciales de IA")
    print(f"  -> Version 1 Guardada. Commit Hash: {commit_v1}")
    
    # 4. Modificar Dataset: Crear Version 2
    # - Modificamos la instrucción 2
    # - Agregamos una nueva instrucción 4
    # - Eliminamos la instrucción 1
    dataset_v2 = [
        {"instruction": "Calcula el embedding de RAG hibrido en PyTorch", "output": "vector = trainer.encode(...)"},  # Modificado
        {"instruction": "Como indexar grafos HNSW?", "output": "HNSW indexa en capas..."},
        {"instruction": "Concepto basico de BPE tokenizer", "output": "BPE segmenta en subpalabras..."}  # Nuevo
    ]
    
    with open(dataset_file, "w", encoding="utf-8") as f:
        json.dump(dataset_v2, f, indent=2, ensure_ascii=False)
        
    print("\n--- PASO 2: Mutando dataset y confirmando Version 2 ---")
    commit_v2 = vcs.commit(dataset_file, "v2: Corregir RAG, añadir BPE y eliminar recursividad")
    print(f"  -> Version 2 Guardada. Commit Hash: {commit_v2}")
    
    # 5. Calcular Diffs de datos
    print("\n--- PASO 3: Calculando diferencias semanticas (Diff) entre v1 y v2 ---")
    diff_report = vcs.diff(commit_v1, commit_v2)
    print(json.dumps(diff_report, indent=2))
    
    # 6. Checkout a Version 1
    print("\n--- PASO 4: Ejecutando checkout para restaurar Version 1 ---")
    vcs.checkout(commit_v1, dataset_file)
    
    # Comprobar el contenido del archivo restaurado
    with open(dataset_file, "r", encoding="utf-8") as f:
        restored = json.load(f)
        
    print(f"  -> Dataset restaurado en disco. Total registros: {len(restored)}")
    for i, item in enumerate(restored, 1):
        print(f"     Registro {i}: '{item['instruction'][:50]}...'")
        
    # Limpiamos el entorno del ejemplo
    if os.path.exists(dataset_file):
        os.remove(dataset_file)
    if os.path.exists(store_dir):
        shutil.rmtree(store_dir)
    print("\nEntorno de demostracion limpiado exitosamente.")


if __name__ == "__main__":
    run_demo()
