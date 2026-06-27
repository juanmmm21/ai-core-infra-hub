import os
import shutil
import torch
import torch.nn as nn
from finetuner import LoraConfig, QLoraPipeline

# Fijamos semillas aleatorias para reproducibilidad en la consola
torch.manual_seed(42)

# Definimos una red simple MLP simulando un codificador de lenguaje o proyeccion
class ToyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 4)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def run_demo() -> None:
    print("=" * 70)
    print("      Demostración de Ajuste Fino Eficiente (QLoRA) con PyTorch       ")
    print("=" * 70)
    
    # 1. Crear modelo base
    print("\n--- PASO 1: Inicializando modelo MLP base y cargando pesos ---")
    model = ToyEncoder()
    print("Modelo base cargado:")
    print(model)
    
    # 2. Configurar e inyectar adaptadores LoRA
    print("\n--- PASO 2: Inyectando adaptadores LoRA en capas fc1 y fc2 ---")
    config = LoraConfig(
        r=4,
        alpha=8.0,
        dropout=0.05,
        target_modules=["fc1", "fc2"]
    )
    pipeline = QLoraPipeline(config)
    
    # Aplicar LoRA con cuantización de 4 bits simulada en la capa base
    pipeline.apply_lora(model, quantize_4bit=True)
    print("\nModelo con adaptadores inyectados (congelado y cuantizado en 4-bit):")
    print(model)
    
    # Mostrar resumen de parámetros
    pipeline.print_trainable_parameters_summary(model)
    
    # 3. Datos sintéticos para entrenamiento
    print("\n--- PASO 3: Generando dataset sintético de entrenamiento ---")
    # Generamos 100 muestras aleatorias
    X_train = torch.randn(100, 16)
    # Definimos una función objetivo simple de mapeo para entrenar el modelo
    Y_target = torch.sin(X_train[:, :4]) * 2.0  # El objetivo es predecir una función no lineal de las primeras 4 dimensiones
    
    # 4. Bucle de entrenamiento del adaptador LoRA
    print("\n--- PASO 4: Iniciando bucle de optimización (50 épocas) ---")
    criterion = nn.MSELoss()
    # Solo pasamos al optimizador los parámetros que requieren gradiente (las matrices A y B de LoRA)
    trainable_params = [p for _, p in pipeline.get_trainable_parameters(model)]
    optimizer = torch.optim.AdamW(trainable_params, lr=0.01)
    
    model.train()
    for epoch in range(1, 51):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, Y_target)
        loss.backward()
        optimizer.step()
        
        if epoch == 1 or epoch % 10 == 0:
            print(f"  Epoca {epoch:02d}/50 | MSE Loss: {loss.item():.6f}")
            
    # 5. Guardado de los pesos adaptadores (PEFT checkpoint)
    print("\n--- PASO 5: Guardando pesos de adaptadores y archivos de configuracion ---")
    checkpoint_dir = "./adapters_checkpoint"
    pipeline.save_adapters(model, checkpoint_dir)
    
    # 6. Carga y verificación del adaptador en un modelo base limpio
    print("\n--- PASO 6: Verificando la restauración en un modelo nuevo ---")
    torch.manual_seed(42)  # Re-sembrar para inicializar con los mismos pesos base exactos
    new_model = ToyEncoder()
    pipeline.apply_lora(new_model, quantize_4bit=True)

    
    # Realizar inferencia antes de cargar los pesos entrenados
    new_model.eval()
    with torch.no_grad():
        pred_before = new_model(X_train[:2])
        
    # Cargar los adaptadores guardados
    pipeline.load_adapters(new_model, checkpoint_dir)
    
    # Realizar inferencia después de cargar los pesos entrenados
    with torch.no_grad():
        pred_after = new_model(X_train[:2])
        
    # El modelo entrenado original
    model.eval()
    with torch.no_grad():
        pred_trained = model(X_train[:2])
        
    print("\nResultados de predicción de muestra (primeros 2 registros):")
    print(f"  Modelo base inicial (antes de cargar adapter):\n  {pred_before}")
    print(f"  Modelo entrenado original:\n  {pred_trained}")
    print(f"  Nuevo modelo con adaptador acoplado:\n  {pred_after}")
    
    # Verificar equivalencia exacta
    diff = torch.max(torch.abs(pred_trained - pred_after)).item()
    print(f"\nDiferencia máxima entre modelo original y modelo restaurado: {diff:.6e}")
    if diff < 1e-6:
        print("-> [EXITO] Los pesos se restauraron y acoplaron con precision matemática absoluta.")
    else:
        print("-> [FALLO] Discrepancia en los resultados decuantizados del adaptador.")
        
    # Limpiar el entorno local
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        print("\nDirectorio temporal de adaptadores eliminado correctamente.")


if __name__ == "__main__":
    run_demo()
