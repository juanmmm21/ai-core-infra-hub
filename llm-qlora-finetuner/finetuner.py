import os
import json
import math
import logging
from typing import Dict, List, Any, Tuple, Optional
import torch
import torch.nn as nn
from pydantic import BaseModel, Field

# Configuración del registrador de eventos
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Representación matemática de los 16 cuantiles del tipo NormalFloat4 (NF4)
# Diseñados por Tim Dettmers et al. para cuantización óptima de pesos con distribución normal.
NF4_QUANTILES = [
    -1.0,
    -0.6961917006969452,
    -0.5250730514526367,
    -0.3949161171913147,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105026721954346,
    0.0,
    0.07958029866218567,
    0.1609302043914795,
    0.24611230194568634,
    0.33791524171829224,
    0.4407000780105591,
    0.5626170039176941,
    0.7229568362236023,
    1.0
]


class LoraConfig(BaseModel):
    """Configuración de parámetros para la adaptación de bajo rango (LoRA)."""
    r: int = Field(default=8, description="Rango de las matrices de adaptacion")
    alpha: float = Field(default=16.0, description="Factor de escala constante para regular la influencia de LoRA")
    target_modules: List[str] = Field(default=["linear"], description="Lista de nombres de modulos a adaptar")
    dropout: float = Field(default=0.05, description="Tasa de dropout aplicada a las activaciones de entrada de LoRA")


class NF4Quantizer:
    """
    Simulador de cuantización NormalFloat4 (NF4) bloque por bloque.
    
    Permite emular la compresión de precisión de 32/16 bits a 4 bits (NF4)
    sin requerir compilaciones complejas de CUDA de bitsandbytes en local.
    """

    @staticmethod
    def quantize(tensor: torch.Tensor, block_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cuantiza un tensor flotante a NF4 de forma bloque-a-bloque.
        
        Args:
            tensor: Tensor a cuantizar.
            block_size: Tamaño del bloque para normalización absmax.
            
        Returns:
            - indices: Tensor del mismo tamaño con valores enteros (0-15) representando indices NF4.
            - scales: Tensor con las escalas absmax correspondientes a cada bloque.
        """
        original_shape = tensor.shape
        flat_tensor = tensor.flatten()
        num_elements = flat_tensor.numel()
        
        # Relleno (padding) si el número de elementos no es divisible por block_size
        padding_size = (block_size - (num_elements % block_size)) % block_size
        if padding_size > 0:
            flat_tensor = torch.cat([flat_tensor, torch.zeros(padding_size, device=tensor.device)])
            
        # Redimensionar en bloques
        blocked_tensor = flat_tensor.view(-1, block_size)
        
        # Calcular el máximo absoluto por bloque (evitamos división por cero con epsilon)
        scales = torch.max(torch.abs(blocked_tensor), dim=1, keepdim=True)[0]
        scales = torch.clamp(scales, min=1e-8)
        
        # Normalizar el contenido del bloque al rango [-1.0, 1.0]
        normalized_blocked = blocked_tensor / scales
        
        # Buscar el cuantil NF4 más cercano para cada elemento
        quantiles = torch.tensor(NF4_QUANTILES, device=tensor.device, dtype=tensor.dtype)
        # Calculamos la distancia absoluta de cada elemento a los 16 cuantiles
        diffs = torch.abs(normalized_blocked.unsqueeze(-1) - quantiles) # Shape: [num_blocks, block_size, 16]
        indices = torch.argmin(diffs, dim=-1).to(torch.uint8) # Shape: [num_blocks, block_size]
        
        # Devolver a la forma original (recortando el padding si existiera)
        indices_flat = indices.flatten()
        if padding_size > 0:
            indices_flat = indices_flat[:-padding_size]
            
        return indices_flat.view(original_shape), scales.flatten()

    @staticmethod
    def dequantize(indices: torch.Tensor, scales: torch.Tensor, original_shape: torch.Size, block_size: int = 64) -> torch.Tensor:
        """
        Decuantiza los índices NF4 de vuelta a valores flotantes usando las escalas guardadas.
        """
        flat_indices = indices.flatten()
        num_elements = flat_indices.numel()
        
        padding_size = (block_size - (num_elements % block_size)) % block_size
        if padding_size > 0:
            flat_indices = torch.cat([flat_indices, torch.zeros(padding_size, device=indices.device, dtype=torch.uint8)])
            
        blocked_indices = flat_indices.view(-1, block_size)
        
        # Mapear índices a los valores reales de cuantiles NF4
        quantiles = torch.tensor(NF4_QUANTILES, device=indices.device, dtype=torch.float32)
        dequantized_blocked = quantiles[blocked_indices.long()] # Shape: [num_blocks, block_size]
        
        # Multiplicar por las escalas de bloque correspondiente
        scales_expanded = scales.unsqueeze(-1) # Shape: [num_blocks, 1]
        reconstructed_blocked = dequantized_blocked * scales_expanded
        
        # Aplanar y recortar padding
        reconstructed_flat = reconstructed_blocked.flatten()
        if padding_size > 0:
            reconstructed_flat = reconstructed_flat[:-padding_size]
            
        return reconstructed_flat.view(original_shape)


class LoraLinear(nn.Module):
    """
    Envoltura (wrapper) LoRA para una capa lineal.
    
    Mantiene la capa lineal original congelada y opcionalmente cuantizada en NF4,
    inyectando un camino alternativo de bajo rango (matrices lora_A y lora_B).
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
        quantize_4bit: bool = True,
        block_size: int = 64
    ) -> None:
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.quantize_4bit = quantize_4bit
        self.block_size = block_size
        
        # Copiamos el sesgo (bias) si existe
        if base_layer.bias is not None:
            self.bias = nn.Parameter(base_layer.bias.data.clone())
        else:
            self.register_parameter('bias', None)
            
        # Si activamos cuantización QLoRA, comprimimos los pesos de la capa base a NF4
        if quantize_4bit:
            indices, scales = NF4Quantizer.quantize(base_layer.weight.data, block_size)
            # Guardamos como buffers para evitar que PyTorch intente calcular gradientes para la capa base
            self.register_buffer("quantized_weight_indices", indices)
            self.register_buffer("quantized_weight_scales", scales)
            self.register_buffer("original_weight_shape", torch.tensor(base_layer.weight.shape))
            logger.info(f"Capa base cuantizada a NF4. Ahorro de memoria simulado: ~75% de VRAM")
        else:
            # Si no se cuantiza, guardamos el peso original y lo congelamos explícitamente
            self.weight = nn.Parameter(base_layer.weight.data.clone(), requires_grad=False)

        # Definición de las matrices de bajo rango adaptadoras (trainables)
        # lora_A se inicializa con Kaiming Uniform para romper simetrías.
        # lora_B se inicializa en ceros, garantizando que el delta (B * A) al inicio sea exactamente 0.
        self.lora_A = nn.Parameter(torch.zeros((r, self.in_features)), requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, r)), requires_grad=True)
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Dropout regulador para las activaciones
        self.lora_dropout = nn.Dropout(p=dropout)

    def get_dequantized_weight(self) -> torch.Tensor:
        """Decuantiza dinámicamente los pesos base de la capa para computar el forward."""
        if self.quantize_4bit:
            shape = torch.Size(self.quantized_weight_shape_list())
            return NF4Quantizer.dequantize(
                self.quantized_weight_indices,
                self.quantized_weight_scales,
                shape,
                self.block_size
            ).to(self.lora_A.dtype)
        return self.weight

    def quantized_weight_shape_list(self) -> List[int]:
        """Obtiene la forma original del peso cuantizado en formato lista compatible con CPU/GPU."""
        return self.original_weight_shape.tolist()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Obtener peso base decuantizado dinámicamente
        w_base = self.get_dequantized_weight()
        
        # 2. Computar proyección base (W_base * x)
        base_out = nn.functional.linear(x, w_base, self.bias)
        
        # 3. Computar adaptación de bajo rango: (x * lora_A^T) * lora_B^T * scaling
        # Aplicamos dropout a la entrada para prevenir co-adaptación
        x_dropped = self.lora_dropout(x)
        lora_out = (x_dropped @ self.lora_A.t()) @ self.lora_B.t()
        lora_out = lora_out * self.scaling
        
        # 4. Combinación aditiva
        return base_out + lora_out


class QLoraPipeline:
    """
    Pipeline unificado para inyectar, entrenar y persistir adaptadores QLoRA.
    """

    def __init__(self, config: LoraConfig) -> None:
        self.config = config

    def _replace_layers(self, model: nn.Module, quantize_4bit: bool) -> None:
        """Helper recursivo para inyectar wrappers LoraLinear."""
        for name, module in list(model.named_children()):
            if isinstance(module, nn.Linear):
                # Comprobamos si el nombre coincide con alguno de los objetivos configurados
                if any(target in name for target in self.config.target_modules):
                    # Crear el reemplazo
                    lora_layer = LoraLinear(
                        base_layer=module,
                        r=self.config.r,
                        alpha=self.config.alpha,
                        dropout=self.config.dropout,
                        quantize_4bit=quantize_4bit
                    )
                    setattr(model, name, lora_layer)
                    logger.info(f"Capa '{name}' adaptada exitosamente con LoRA (r={self.config.r}, alpha={self.config.alpha})")
            else:
                # Recursión para submódulos jerárquicos
                self._replace_layers(module, quantize_4bit)

    def apply_lora(self, model: nn.Module, quantize_4bit: bool = True) -> nn.Module:
        """
        Reemplaza recursivamente las capas nn.Linear objetivo por capas LoraLinear adaptadas
        y congela todos los parámetros del modelo base para asegurar que solo se entrenen
        las matrices lora_A y lora_B.
        """
        self._replace_layers(model, quantize_4bit)
        
        # Congelar todos los parámetros
        for param in model.parameters():
            param.requires_grad = False
            
        # Descongelar únicamente las matrices de adaptador lora_A y lora_B
        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                param.requires_grad = True
                
        return model

    def get_trainable_parameters(self, model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
        """Devuelve únicamente los parámetros que tienen activado el cálculo de gradiente."""
        trainables = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainables.append((name, param))
        return trainables

    def print_trainable_parameters_summary(self, model: nn.Module) -> None:
        """Muestra un resumen analítico del porcentaje de parámetros entrenables en el modelo."""
        total_params = 0
        trainable_params = 0
        for _, param in model.named_parameters():
            # Si el modelo tiene pesos cuantizados en buffers, sumamos su tamaño correspondiente
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                
        # Contabilizar el tamaño de los pesos de la capa base cuantizada almacenada en buffers
        for _, buf in model.named_buffers():
            if "quantized_weight" in _ or "original_weight_shape" in _:
                # Evitamos duplicados, solo contamos los índices cuantizados como representativos
                if "indices" in _:
                    total_params += buf.numel()

        percentage = 100 * trainable_params / max(total_params, 1)
        logger.info(f"Parámetros entrenables: {trainable_params:,} | Parámetros totales: {total_params:,} | Ratio: {percentage:.4f}%")

    def save_adapters(self, model: nn.Module, output_dir: str) -> str:
        """
        Guarda únicamente las matrices de adaptador entrenadas y su configuración.
        
        Esto demuestra el beneficio de PEFT, donde el checkpoint resultante pesa unos pocos KB.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Extraer estado del adaptador
        adapter_state_dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad and ("lora_A" in name or "lora_B" in name):
                adapter_state_dict[name] = param.data.cpu()
                
        # Guardar pesos
        weights_path = os.path.join(output_dir, "adapter_model.bin")
        torch.save(adapter_state_dict, weights_path)
        
        # Guardar configuración
        config_path = os.path.join(output_dir, "adapter_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.model_dump(), f, indent=2)
            
        logger.info(f"Adaptadores guardados en '{output_dir}'. Tamaño estimado: {os.path.getsize(weights_path) / 1024:.2f} KB")
        return output_dir

    def load_adapters(self, model: nn.Module, adapter_dir: str) -> None:
        """
        Carga y acopla las matrices del adaptador guardadas en un modelo con wrappers LoRA.
        """
        weights_path = os.path.join(adapter_dir, "adapter_model.bin")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Archivo de pesos del adaptador '{weights_path}' no encontrado.")
            
        adapter_state_dict = torch.load(weights_path)
        
        # Cargar los parámetros en el modelo
        model_state = model.state_dict()
        for name, param in adapter_state_dict.items():
            if name in model_state:
                # Comprobar que coincida el tamaño
                if model_state[name].shape == param.shape:
                    model_state[name].copy_(param)
                    logger.info(f"Parámetro adaptador '{name}' cargado correctamente.")
                else:
                    raise ValueError(f"Error de dimensiones para '{name}': {model_state[name].shape} vs {param.shape}")
            else:
                logger.warning(f"Parámetro '{name}' no coincide con la estructura del modelo actual.")
