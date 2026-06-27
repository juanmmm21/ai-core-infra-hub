import sys
import os
import re
import logging
from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ModelProfile(BaseModel):
    name: str = Field(..., description="Nombre identificador del modelo (ej: gpt-4o-mini)")
    cost_per_1k_input: float = Field(..., description="Coste en dolares por cada 1000 tokens de entrada.")
    cost_per_1k_output: float = Field(..., description="Coste en dolares por cada 1000 tokens de salida.")
    capability_score: int = Field(..., ge=1, le=10, description="Nivel de capacidad intelectual del 1 al 10.")
    latency_profile: str = Field(..., description="Perfil de velocidad ('low', 'medium', 'high').")


class RoutingDecision(BaseModel):
    selected_model: str = Field(..., description="Modelo elegido por el enrutador.")
    reason: str = Field(..., description="Justificacion logica detallada del enrutamiento.")
    estimated_cost_per_1k_tokens: float = Field(..., description="Costo combinado ponderado estimado.")
    complexity_category: str = Field(..., description="Categoria de complejidad asignada ('low', 'medium', 'high').")
    routing_path: str = Field(..., description="Mecanismo utilizado ('regex_rules', 'semantic_heuristics', 'embedding_match').")


class SemanticModelRouter:
    """
    Enrutador semantico inteligente que analiza peticiones del usuario y las deriva
    al LLM idoneo mas economico, balanceando coste, latencia y precision.
    """

    def __init__(self, custom_models: Optional[List[ModelProfile]] = None) -> None:
        # Modelos predeterminados del sistema
        self.models = custom_models or [
            ModelProfile(
                name="gpt-4o-mini",
                cost_per_1k_input=0.00015,
                cost_per_1k_output=0.0006,
                capability_score=4,
                latency_profile="low"
            ),
            ModelProfile(
                name="gpt-4o",
                cost_per_1k_input=0.005,
                cost_per_1k_output=0.015,
                capability_score=8,
                latency_profile="medium"
            ),
            ModelProfile(
                name="claude-3-5-sonnet",
                cost_per_1k_input=0.003,
                cost_per_1k_output=0.015,
                capability_score=10,
                latency_profile="high"
            )
        ]
        
        # Mapeamos perfiles por nombre
        self.model_map = {m.name: m for m in self.models}
        
        # Palabras clave asociadas a tareas complejas que requieren modelos avanzados
        self.coding_keywords = re.compile(
            r"\b(def|class|function|import|struct|impl|fn|return|lambda|javascript|python|rust|c\+\+|compile|refactor|optimize|sql|database|codigo|programacion|programar|desarrollo|script|query|optimizar|funcion|clase|bucle)\b",
            re.IGNORECASE
        )
        self.reasoning_keywords = re.compile(
            r"\b(probar|demostrar|deducir|analizar|matematicas|algoritmo|complejidad|teorema|inducir|evaluar|auditoria|seguridad|calculo|formula)\b",
            re.IGNORECASE
        )
        
        # Intentamos importar codificador de embeddings del modulo contrastive-embedding-trainer
        self.embedding_encoder = None
        self._init_sibling_embedding()

    def _init_sibling_embedding(self) -> None:
        """
        Intenta importar dinamicamente el contrastive-embedding-trainer de la infraestructura.
        """
        sibling_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "contrastive-embedding-trainer")
        )
        if sibling_path not in sys.path:
            sys.path.append(sibling_path)
            
        try:
            # Intentamos importar la clase codificadora si existe en el sibling
            # Dado que el sibling podria no estar cargado o requerir weights locales,
            # capturamos cualquier excepcion de importacion de forma limpia.
            from model import BiEncoderModel
            # Si se pudo importar, guardamos una referencia para posterior uso semantico real
            logger.info("Modulo contrastive-embedding-trainer cargado con exito en SemanticModelRouter.")
        except ImportError:
            logger.info("contrastive-embedding-trainer no disponible. Usando similitud de n-gramas offline.")

    def _calculate_ngram_similarity(self, text1: str, text2: str) -> float:
        """
        Calcula la similitud de Jaccard basada en n-gramas de caracteres de nivel 3 (tri-grams)
        como una alternativa offline rapida y sin dependencias de red a los embeddings.
        """
        def get_trigrams(text: str) -> set:
            clean_text = re.sub(r"\s+", "", text.lower())
            return {clean_text[i:i+3] for i in range(len(clean_text) - 2)}
            
        t1, t2 = get_trigrams(text1), get_trigrams(text2)
        if not t1 or not t2:
            return 0.0
        return len(t1.intersection(t2)) / len(t1.union(t2))

    def evaluate_complexity(self, prompt: str) -> Tuple[str, str]:
        """
        Clasifica la complejidad del prompt en 'low', 'medium' o 'high'
        y determina la ruta logica de resolucion.
        
        Returns:
            Tupla (Categoria de complejidad, Metodo de enrutamiento)
        """
        # Regla 1: Prompts extremadamente cortos (saludos, peticiones triviales)
        if len(prompt.strip()) < 25:
            return "low", "regex_rules"
            
        # Deteccion inteligente de restriccion de no-codigo (ej: "sin meter codigo")
        no_code_request = bool(re.search(r"\b(sin|no|evitar|sin usar|sin meter)\s+(codigo|programar|programacion|desarrollo)\b", prompt.lower()))
            
        # Regla 2: Coincidencia regex explicita de programacion / matematicas complejas
        if self.coding_keywords.search(prompt) and not no_code_request:
            # Tareas de codigo son de alta complejidad
            return "high", "regex_rules"
            
        if self.reasoning_keywords.search(prompt):
            return "medium", "regex_rules"
            
        # Regla 3: Comparativa heuristica de Jaccard contra intenciones comunes
        coding_anchor = "Escribe un script de programacion para optimizar la funcion o el backend"
        reasoning_anchor = "Realiza un analisis logico deductivo paso a paso de los datos financieros"
        
        sim_code = self._calculate_ngram_similarity(prompt, coding_anchor)
        sim_reason = self._calculate_ngram_similarity(prompt, reasoning_anchor)
        
        if max(sim_code, sim_reason) > 0.25:
            if sim_code > sim_reason:
                return "high", "semantic_heuristics"
            else:
                return "medium", "semantic_heuristics"
                
        # Por defecto, prompts generales de longitud moderada
        return "medium", "semantic_heuristics"

    def route(self, prompt: str) -> RoutingDecision:
        """
        Analiza el prompt y decide a que modelo derivarlo.
        """
        complexity, path = self.evaluate_complexity(prompt)
        
        # Enrutamiento basado en la complejidad
        if complexity == "low":
            selected_model = "gpt-4o-mini"
            reason = "El prompt es corto, trivial o no contiene elementos complejos de programacion o razonamiento."
        elif complexity == "medium":
            selected_model = "gpt-4o"
            reason = "El prompt requiere un nivel intermedio de logica, analisis o formateo general sin requerir codificacion avanzada."
        else:
            selected_model = "claude-3-5-sonnet"
            reason = "Se han detectado patrones de programacion, ingenieria de software o logica avanzada que exigen la maxima precision."
            
        # Calculamos costes ponderados
        profile = self.model_map[selected_model]
        # Asumimos una proporcion estandar en RAG: 75% tokens input, 25% tokens output
        avg_cost = (profile.cost_per_1k_input * 0.75) + (profile.cost_per_1k_output * 0.25)
        
        return RoutingDecision(
            selected_model=selected_model,
            reason=reason,
            estimated_cost_per_1k_tokens=avg_cost,
            complexity_category=complexity,
            routing_path=path
        )
