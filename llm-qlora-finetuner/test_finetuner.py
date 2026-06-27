import unittest
import os
import shutil
import tempfile
import torch
import torch.nn as nn
from finetuner import LoraConfig, LoraLinear, QLoraPipeline, NF4Quantizer

class TestQLoraFinetuner(unittest.TestCase):

    def setUp(self) -> None:
        # Fijar semilla aleatoria para reproducibilidad matemática
        torch.manual_seed(42)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_nf4_quantization_dequantization(self) -> None:
        """Verifica que la simulacion de cuantizacion/decuantizacion NF4 preserva la estructura."""
        # Creamos un tensor simulando pesos inicializados normalmente
        tensor = torch.randn(10, 20)
        
        # Cuantizar a NF4
        indices, scales = NF4Quantizer.quantize(tensor, block_size=10)
        self.assertEqual(indices.shape, tensor.shape)
        # 200 elementos divididos en bloques de 10 = 20 escalas
        self.assertEqual(scales.shape[0], 20)
        
        # Decuantizar
        reconstructed = NF4Quantizer.dequantize(indices, scales, tensor.shape, block_size=10)
        self.assertEqual(reconstructed.shape, tensor.shape)
        
        # Verificar que el error cuadratico medio es acotado (cuantizacion con perdida pero razonable)
        mse = torch.mean((tensor - reconstructed) ** 2).item()
        self.assertLess(mse, 0.25, "El error de cuantización NF4 excede el límite razonable")

    def test_lora_linear_parameters_and_gradients(self) -> None:
        """Verifica que solo los pesos del adaptador LoRA computan gradientes."""
        base_linear = nn.Linear(10, 5)
        lora_layer = LoraLinear(base_linear, r=4, alpha=8.0, quantize_4bit=True)
        
        # Verificar congelamiento de pesos base
        # Como está cuantizada, los pesos base están en buffers y no admiten gradients
        self.assertFalse(has_name_in_parameters(lora_layer, "weight"))
        self.assertTrue(hasattr(lora_layer, "quantized_weight_indices"))
        
        # Verificar que lora_A y lora_B admiten gradientes
        self.assertTrue(lora_layer.lora_A.requires_grad)
        self.assertTrue(lora_layer.lora_B.requires_grad)
        
        # Ejecutar forward y backward
        x = torch.randn(2, 10)
        out = lora_layer(x)
        self.assertEqual(out.shape, (2, 5))
        
        loss = out.sum()
        loss.backward()
        
        # Verificar flujos de gradientes
        self.assertIsNotNone(lora_layer.lora_A.grad)
        self.assertIsNotNone(lora_layer.lora_B.grad)

    def test_pipeline_integration_and_serialization(self) -> None:
        """Verifica la inyeccion automatica y persistencia de pesos."""
        # Creamos una red neuronal ficticia
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(8, 8)
                self.linear2 = nn.Linear(8, 4)
            def forward(self, x):
                return self.linear2(torch.relu(self.linear1(x)))
                
        model = ToyModel()
        config = LoraConfig(r=2, alpha=4.0, target_modules=["linear1"])
        pipeline = QLoraPipeline(config)
        
        # Aplicar LoRA
        pipeline.apply_lora(model, quantize_4bit=True)
        
        # Verificar que linear1 es LoraLinear y linear2 sigue siendo Linear estándar
        self.assertTrue(isinstance(model.linear1, LoraLinear))
        self.assertTrue(isinstance(model.linear2, nn.Linear))
        
        # Verificar parámetros entrenables
        trainable_params = pipeline.get_trainable_parameters(model)
        names = [name for name, _ in trainable_params]
        # Deberian estar unicamente lora_A, lora_B y bias de linear1 (el bias es opcionalmente entrenable)
        self.assertIn("linear1.lora_A", names)
        self.assertIn("linear1.lora_B", names)
        self.assertNotIn("linear2.weight", names)
        
        # Realizar entrenamiento sintetico de un paso
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        x = torch.randn(4, 8)
        y = torch.randn(4, 4)
        
        # Guardar valor de peso inicial
        initial_lora_A = model.linear1.lora_A.clone()
        
        # Realizar entrenamiento sintetico de varios pasos para actualizar lora_A
        for _ in range(5):
            pred = model(x)
            loss = nn.MSELoss()(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        
        # Validar actualizacion del parametro
        self.assertFalse(torch.equal(initial_lora_A, model.linear1.lora_A))
        
        # Guardar adaptadores
        pipeline.save_adapters(model, self.temp_dir)
        
        # Crear un modelo nuevo idéntico
        new_model = ToyModel()
        pipeline.apply_lora(new_model, quantize_4bit=True)
        
        # Asegurar que inicialmente son diferentes
        self.assertFalse(torch.equal(model.linear1.lora_A, new_model.linear1.lora_A))
        
        # Cargar adaptadores en el nuevo modelo
        pipeline.load_adapters(new_model, self.temp_dir)
        
        # Asegurar que se cargaron exitosamente igualando los pesos entrenados
        self.assertTrue(torch.equal(model.linear1.lora_A, new_model.linear1.lora_A))
        self.assertTrue(torch.equal(model.linear1.lora_B, new_model.linear1.lora_B))


def has_name_in_parameters(module: nn.Module, param_name: str) -> bool:
    """Helper para verificar si un parametro existe en el modulo."""
    for name, _ in module.named_parameters():
        if name == param_name:
            return True
    return False


if __name__ == '__main__':
    unittest.main()
