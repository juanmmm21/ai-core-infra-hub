import sys
import os
import time
import math
import uuid
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

# Intentamos importar la base de datos del sibling nano-vector-db
NANO_DB_AVAILABLE = False
NanoVectorDB = None

try:
    sibling_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "nano-vector-db")
    )
    if sibling_path not in sys.path:
        sys.path.append(sibling_path)
    # Suponemos que en nano-vector-db/database.py esta la clase principal
    from database import NanoVectorDB
    NANO_DB_AVAILABLE = True
except ImportError:
    pass


class AgenticMemory:
    """
    Capa de memoria persistente y estructurada para agentes de Inteligencia Artificial.
    
    Combina:
    1. Memoria de Corto Plazo (Conversational Buffer): Almacena las ultimas interacciones
       aplicando un algoritmo de sintesis/resumen automatico al sobrepasar limites.
    2. Memoria de Largo Plazo (Episodic Decay Memory): Guarda hechos persistentes con
       puntuaciones de importancia y formula de decaimiento logaritmico-temporal de olvido.
    """

    def __init__(
        self,
        short_term_max_turns: int = 5,
        decay_factor: float = 0.05,
        vector_dim: int = 64
    ) -> None:
        self.max_turns = short_term_max_turns
        self.decay_factor = decay_factor
        self.vector_dim = vector_dim
        
        # Buffer de corto plazo
        self.short_term: List[Dict[str, str]] = []
        self.conversation_summary: str = ""
        
        # Almacenamiento de largo plazo (Base de Datos Vectorial local)
        self.db_path = "agent_long_term_memory.json"
        self.use_real_db = NANO_DB_AVAILABLE
        self.vector_db = None
        
        # Fallback local in-memory
        self.fallback_db: List[Dict[str, Any]] = []
        
        if self.use_real_db:
            try:
                # Inicializamos la base de datos vectorial real del modulo hermano
                if os.path.exists(self.db_path):
                    self.vector_db = NanoVectorDB.load(self.db_path)
                else:
                    self.vector_db = NanoVectorDB(dimension=self.vector_dim)
                logger.info("Base de datos vectorial NanoVectorDB conectada exitosamente a la memoria del agente.")
            except Exception as e:
                logger.warning(f"Error al inicializar NanoVectorDB: {str(e)}. Usando fallback de lista en memoria.")
                self.use_real_db = False

    def add_interaction(self, role: str, content: str) -> None:
        """Añade un turno al buffer de corto plazo y gatilla resumen si excede max_turns."""
        self.short_term.append({"role": role, "content": content})
        
        # Si excedemos el limite de turnos (par de usuario-asistente cuenta como 2 turnos)
        if len(self.short_term) > self.max_turns * 2:
            self._summarize_short_term()

    def _summarize_short_term(self) -> None:
        """
        Sintetiza la conversacion acumulada en un resumen para liberar espacio de contexto.
        En produccion, esto llamaria a un LLM. Aqui implementamos un extractor secuencial de eventos.
        """
        turns_to_summarize = self.short_term[:-2]  # Dejamos las ultimas 2 interacciones intactas
        self.short_term = self.short_term[-2:]
        
        # Compilar resumen heuristico
        extracted_facts = []
        for turn in turns_to_summarize:
            content = turn["content"]
            # Extraemos afirmaciones clave simples
            if turn["role"] == "user" and len(content) > 10:
                extracted_facts.append(f"El usuario pregunto o menciono: '{content}'")
            elif turn["role"] == "assistant" and len(content) > 10:
                extracted_facts.append(f"El agente respondio: '{content[:50]}...'")
                
        # Unimos al resumen historico
        prefix = f"{self.conversation_summary}\n" if self.conversation_summary else ""
        self.conversation_summary = prefix + " -> " + " | ".join(extracted_facts)
        logger.info(f"Memoria a corto plazo resumida. Historial acumulado: {len(self.conversation_summary)} caracteres.")

    def _generate_mock_embedding(self, text: str) -> List[float]:
        """
        Genera un vector unitario deterministicamente basado en una bolsa de palabras 
        hasheadas con MD5, garantizando alta similitud solo ante coincidencia de terminos.
        """
        import hashlib
        import re
        words = re.findall(r"\w+", text.lower())
        vector = [0.0] * self.vector_dim
        if not words:
            return vector
            
        for w in words:
            h_val = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
            h = h_val % self.vector_dim
            vector[h] += 1.0
            
        # Normalizacion L2 para similitud unitaria directa
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector

    def save_fact(self, fact: str, importance: int = 5) -> None:
        """
        Persiste un hecho relevante en la memoria episódica de largo plazo.
        
        Args:
            fact: Hecho a memorizar.
            importance: Calificacion de relevancia emocional o pragmatica (1 a 10).
        """
        embedding = self._generate_mock_embedding(fact)
        timestamp = time.time()
        
        fact_item = {
            "fact": fact,
            "importance": importance,
            "created_at": timestamp,
            "last_accessed": timestamp,
            "embedding": embedding
        }
        
        if self.use_real_db and self.vector_db is not None:
            try:
                # Insertamos en el NanoVectorDB
                self.vector_db.insert(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    metadata={"fact": fact, "importance": importance, "created_at": timestamp, "last_accessed": timestamp}
                )
            except Exception as e:
                logger.error(f"Fallo al insertar en NanoVectorDB: {str(e)}. Agregando a fallback local.")
                self.fallback_db.append(fact_item)
        else:
            self.fallback_db.append(fact_item)
            
        logger.info(f"Hecho guardado en memoria de largo plazo (Importancia: {importance}): '{fact}'")

    def recall(self, query: str, top_k: int = 3, current_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Recupera los top_k recuerdos mas relevantes aplicando la formula de decaimiento temporal (olvido).
        
        Formula de puntuacion de recuerdo:
            Score = Similitud_Coseno * Importancia * e^(-lambda * delta_t)
            
        Donde:
            delta_t: Tiempo transcurrido en segundos/unidades temporales.
            lambda: decay_factor (rapidez de olvido).
        """
        if current_time is None:
            current_time = time.time()
            
        query_vector = self._generate_mock_embedding(query)
        candidates: List[Tuple[float, Dict[str, Any]]] = []
        
        # Leemos elementos a calificar
        items = []
        if self.use_real_db and self.vector_db is not None:
            try:
                # Buscamos en el NanoVectorDB usando similitud de coseno
                results = self.vector_db.query(vector=query_vector, top_k=20)
                for res in results:
                    meta = res["metadata"]
                    items.append({
                        "fact": meta["fact"],
                        "importance": meta["importance"],
                        "created_at": meta["created_at"],
                        "last_accessed": meta["last_accessed"],
                        "embedding": res["vector"]
                    })
            except Exception as e:
                logger.error(f"Fallo en consulta a NanoVectorDB: {str(e)}. Usando fallback local.")
                items = self.fallback_db
        else:
            items = self.fallback_db
            
        # Puntuamos cada hecho segun similitud y antiguedad
        for item in items:
            # Similitud de coseno (dot product ya que ambos estan normalizados)
            similarity = sum(qv * iv for qv, iv in zip(query_vector, item["embedding"]))
            
            # Ajustamos rango de similitud de [-1, 1] a [0, 1]
            similarity = (similarity + 1.0) / 2.0
            
            # Calculamos delta de tiempo transcurrido
            delta_t = max(0.0, current_time - item["created_at"])
            
            # Formula de decaimiento exponencial de Ebbinghaus (olvido temporal)
            temporal_decay = math.exp(-self.decay_factor * delta_t)
            
            # Puntuacion final integrada
            recall_score = similarity * item["importance"] * temporal_decay
            
            candidates.append((recall_score, item))
            
        # Ordenamos descendente por puntuacion final
        sorted_candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
        
        # Formateamos resultados de retorno actualizando el timestamp de acceso para refrescar su memoria
        recalled_items = []
        for score, item in sorted_candidates[:top_k]:
            item["last_accessed"] = current_time
            recalled_items.append({
                "fact": item["fact"],
                "importance": item["importance"],
                "recall_score": float(score),
                "created_at": item["created_at"]
            })
            
        return recalled_items
