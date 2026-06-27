import random
import re
import json
import logging
from typing import List, Dict, Any, Set, Tuple, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class InstructionSample(BaseModel):
    instruction: str = Field(..., description="El prompt o instrucción de entrenamiento.")
    output: str = Field(..., description="La respuesta esperada del modelo.")


class DPOSample(BaseModel):
    prompt: str = Field(..., description="El prompt original de entrada.")
    chosen: str = Field(..., description="La respuesta preferida de alta calidad.")
    rejected: str = Field(..., description="La respuesta rechazada con alucinaciones o errores.")


class SyntheticDataGenerator:
    """
    Motor de generacion de datos sinteticos estructurados.
    
    Genera datasets de ajuste fino (Instruction Tuning / DPO) aplicando
    mutaciones semanticas de plantillas (Self-Instruct) y filtros estrictos
    de calidad y diversidad mediante desduplicacion por similitud de n-gramas.
    """

    def __init__(
        self,
        min_length: int = 20,
        max_length: int = 1000,
        dedup_threshold: float = 0.50
    ) -> None:
        self.min_length = min_length
        self.max_length = max_length
        self.dedup_threshold = dedup_threshold
        
        # Diccionario de sinonimos y plantillas tecnicas para la mutacion offline
        self.action_verbs = ["escribe", "desarrolla", "crea", "programa", "implementa", "optimiza"]
        self.topics = {
            "RAG": ["arquitectura RAG", "generacion aumentada por recuperacion", "RAG hibrido", "pipeline RAG"],
            "embeddings": ["vectores de alta dimension", "embeddings semanticos", "embeddings contrastivos"],
            "db": ["base de datos vectorial NanoVectorDB", "indice HNSW", "similitud de coseno"]
        }
        self.modifiers = [
            "en lenguaje Python limpio y con tipado estricto",
            "detallando las decisiones de diseño arquitectonico",
            "con comentarios explicativos paso a paso en español",
            "diseñado para entornos de produccion escalables"
        ]

    def _get_trigrams(self, text: str) -> Set[str]:
        """Extrae tri-gramas a nivel de caracteres para calculo de diversidad."""
        clean = re.sub(r"\s+", "", text.lower())
        if len(clean) < 3:
            return {clean}
        return {clean[i:i+3] for i in range(len(clean) - 2)}

    def calculate_overlap(self, text1: str, text2: str) -> float:
        """Calcula la similitud de Jaccard de tri-gramas entre dos textos."""
        tg1, tg2 = self._get_trigrams(text1), self._get_trigrams(text2)
        if not tg1 or not tg2:
            return 0.0
        return len(tg1.intersection(tg2)) / len(tg1.union(tg2))

    def _mutate_prompt(self, base_topic: str) -> str:
        """Aplica mutaciones linguisticas aleatorias (Self-Instruct style) para diversificar."""
        verb = random.choice(self.action_verbs)
        
        # Recuperamos palabras clave del topico o usamos defaults
        topic_synonyms = self.topics.get(base_topic, [base_topic])
        synonym = random.choice(topic_synonyms)
        modifier = random.choice(self.modifiers)
        
        # Mutamos el formato estructural de la pregunta
        structures = [
            f"{verb.capitalize()} un tutorial sobre {synonym} {modifier}.",
            f"¿Como se {verb} un modulo de {synonym}? Escribe un ejemplo {modifier}.",
            f"Necesito que {verb}s una solucion de {synonym} {modifier}.",
            f"Explicacion y guia practica para {verb} {synonym} {modifier}."
        ]
        
        return random.choice(structures)

    def _generate_synthetic_output(self, instruction: str) -> Tuple[str, str]:
        """
        Sintetiza la respuesta elegida (chosen) y la respuesta rechazada (rejected)
        para DPO basadas en la instruccion mutada.
        """
        instruction_lower = instruction.lower()
        
        # Base de respuestas sinteticas de alta y baja calidad
        if "rag" in instruction_lower:
            chosen = (
                "La arquitectura RAG integra busqueda léxica (BM25) con busqueda vectorial densa. "
                "Para implementarlo en Python, primero cargamos los chunks, extraemos embeddings "
                "mediante contrastive-embedding-trainer y realizamos una recuperacion hibrida fusionada "
                "por RRF antes de enviar el contexto consolidado al LLM."
            )
            rejected = (
                "RAG significa base de datos. Para usar RAG en Python solo necesitas importar la libreria "
                "rag-server y llamar a rag.search() que te devuelve de inmediato la respuesta redactada "
                "del modelo sin necesidad de embeddings ni base de datos vectorial."
            )
        elif "db" in instruction_lower or "nanovectordb" in instruction_lower or "hnsw" in instruction_lower:
            chosen = (
                "NanoVectorDB es una base de datos vectorial en memoria optimizada en Python. "
                "Utiliza el algoritmo HNSW (Hierarchical Navigable Small World) para busquedas de "
                "vecinos mas cercanos aproximados en tiempo sub-lineal, calculando similitud de coseno "
                "y filtrando metadatos estructurados."
            )
            rejected = (
                "NanoVectorDB guarda archivos en carpetas de texto plano llamadas database.txt. "
                "Para buscar un vector, el sistema lee secuencialmente cada linea calculando restas basicas, "
                "lo cual tarda varias horas y no requiere ningun tipo de indexacion HNSW ni optimizaciones."
            )
        else:
            chosen = (
                "Implementacion modular del pipeline de infraestructura de Inteligencia Artificial. "
                "El codigo cuenta con tipado estricto, gestion de excepciones explícitas y comentarios "
                "detallados en español justificando las optimizaciones de sistema realizadas."
            )
            rejected = (
                "Aqui esta el codigo del script: # TODO: implementar logica. No se manejan excepciones "
                "ni se incluye documentacion tecnica adicional."
            )
            
        return chosen, rejected

    def generate_instruction_dataset(self, topics: List[str], count_per_topic: int = 5) -> List[InstructionSample]:
        """Genera un conjunto de datos sinteticos para ajuste fino de instrucciones (Instruction Tuning)."""
        dataset: List[InstructionSample] = []
        existing_prompts: List[str] = []
        
        for topic in topics:
            generated_for_topic = 0
            attempts = 0
            
            # Intentamos generar hasta completar el count respetando filtros de desduplicacion
            while generated_for_topic < count_per_topic and attempts < count_per_topic * 5:
                attempts += 1
                prompt = self._mutate_prompt(topic)
                
                # Chequeo de longitud minima
                if len(prompt) < self.min_length:
                    continue
                    
                # Chequeo de desduplicacion de n-gramas
                is_duplicate = False
                for existing in existing_prompts:
                    if self.calculate_overlap(prompt, existing) > self.dedup_threshold:
                        is_duplicate = True
                        break
                        
                if is_duplicate:
                    continue
                    
                # Generamos respuesta elegida
                chosen, _ = self._generate_synthetic_output(prompt)
                
                # Validacion de longitud de respuesta
                if not (self.min_length <= len(chosen) <= self.max_length):
                    continue
                    
                dataset.append(InstructionSample(instruction=prompt, output=chosen))
                existing_prompts.append(prompt)
                generated_for_topic += 1
                
            logger.info(f"Topico '{topic}': Generados {generated_for_topic} ejemplos unicos (Intentos: {attempts}).")
            
        return dataset

    def generate_dpo_dataset(self, topics: List[str], count_per_topic: int = 5) -> List[DPOSample]:
        """Genera un conjunto de datos sinteticos estructurado para DPO (Chosen vs Rejected)."""
        dataset: List[DPOSample] = []
        existing_prompts: List[str] = []
        
        for topic in topics:
            generated_for_topic = 0
            attempts = 0
            
            while generated_for_topic < count_per_topic and attempts < count_per_topic * 5:
                attempts += 1
                prompt = self._mutate_prompt(topic)
                
                if len(prompt) < self.min_length:
                    continue
                    
                is_duplicate = False
                for existing in existing_prompts:
                    if self.calculate_overlap(prompt, existing) > self.dedup_threshold:
                        is_duplicate = True
                        break
                        
                if is_duplicate:
                    continue
                    
                chosen, rejected = self._generate_synthetic_output(prompt)
                
                if not (self.min_length <= len(chosen) <= self.max_length):
                    continue
                    
                dataset.append(DPOSample(prompt=prompt, chosen=chosen, rejected=rejected))
                existing_prompts.append(prompt)
                generated_for_topic += 1
                
            logger.info(f"Topico DPO '{topic}': Generados {generated_for_topic} ejemplos unicos (Intentos: {attempts}).")
            
        return dataset
