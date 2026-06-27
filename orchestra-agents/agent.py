import re
import logging
from typing import Dict, Any, Callable, List, Tuple, Optional

logger = logging.getLogger(__name__)


class AutonomousAgent:
    """
    Agente autonomo que implementa el bucle de razonamiento ReAct (Thought-Action-Observation).
    
    Permite acoplar herramientas de Python ejecutables, mantiene una bitacora (scratchpad)
    de pensamientos y observaciones, y cuenta con simulacion cognitiva offline
    para autoevaluacion de flujo de control sin dependencias de red o modelos.
    """

    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        tools: Optional[Dict[str, Callable[[str], str]]] = None,
        use_mock_llm: bool = True
    ) -> None:
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.tools = tools or {}
        self.use_mock_llm = use_mock_llm
        self.scratchpad: List[Tuple[str, str, str]] = []  # List of (Thought, Action, Observation)
        
    def add_tool(self, name: str, func: Callable[[str], str]) -> None:
        """Añade una herramienta ejecutable al catalogo del agente."""
        self.tools[name] = func

    def _parse_react_output(self, llm_output: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
        """
        Parsea la salida del LLM bajo el formato estructurado:
        Thought: <pensamiento>
        Action: <nombre_herramienta>(<argumento>)
        Final Answer: <respuesta_final>
        """
        thought = ""
        action_name = None
        action_arg = None
        final_answer = None
        
        # Extraemos Thought
        thought_match = re.search(r"Thought:\s*(.*?)(?=(?:Action:|Final Answer:|$))", llm_output, re.DOTALL | re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()
            
        # Extraemos Action (ej: search_db("HNSW index"))
        action_match = re.search(r"Action:\s*(\w+)\((.*?)\)", llm_output, re.IGNORECASE)
        if action_match:
            action_name = action_match.group(1).strip()
            action_arg = action_match.group(2).strip().strip("\"'")
            
        # Extraemos Final Answer
        final_match = re.search(r"Final Answer:\s*(.*)", llm_output, re.DOTALL | re.IGNORECASE)
        if final_match:
            final_answer = final_match.group(1).strip()
            
        return thought, action_name, action_arg, final_answer

    def execute_react_step(self, task: str) -> Tuple[bool, str]:
        """
        Ejecuta un paso del bucle de razonamiento ReAct.
        
        Returns:
            Tupla (is_finished: bool, step_description: str)
        """
        # 1. Obtener la salida de razonamiento del LLM (simulada o real)
        if self.use_mock_llm:
            llm_output = self._simulate_llm_response(task)
        else:
            # En un entorno real, aqui se formatea la historia, system prompt y scratchpad,
            # y se hace la llamada al servidor de inferencia o API remota.
            llm_output = "Thought: Necesito buscar mas datos.\nFinal Answer: Completado."
            
        # 2. Parsear la respuesta
        thought, action, arg, final_answer = self._parse_react_output(llm_output)
        
        # 3. Guardar pensamiento
        step_desc = f"Pensamiento: {thought}"
        
        if final_answer:
            # El agente ha llegado a una conclusion final
            self.scratchpad.append((thought, "None", f"Final Answer: {final_answer}"))
            return True, f"{step_desc} -> Respuesta Final: {final_answer}"
            
        if action:
            # El agente decide usar una herramienta
            step_desc += f" | Accion: {action}({arg})"
            
            if action in self.tools:
                try:
                    # Ejecutamos la herramienta de forma sincrona y capturamos la observacion
                    observation = self.tools[action](arg)
                except Exception as e:
                    observation = f"Error al ejecutar herramienta: {str(e)}"
            else:
                observation = f"Error: La herramienta '{action}' no esta registrada."
                
            step_desc += f" -> Observacion: {observation}"
            self.scratchpad.append((thought, f"{action}({arg})", observation))
            return False, step_desc
            
        # Si no hay accion ni respuesta final (salida corrupta), forzamos parada
        self.scratchpad.append((thought, "None", "Error: Salida de modelo no concluyente."))
        return True, f"{step_desc} -> Error: Fin abrupto del razonamiento."

    def run(self, task: str, max_steps: int = 5) -> str:
        """
        Ejecuta el bucle ReAct completo de forma autonoma hasta terminar
        o alcanzar el limite de pasos de seguridad.
        """
        self.scratchpad.clear()
        logger.info(f"Agente [{self.name}] iniciando tarea: '{task}'")
        
        for step in range(1, max_steps + 1):
            is_finished, desc = self.execute_react_step(task)
            logger.info(f"Paso {step}: {desc}")
            if is_finished:
                # Retornamos el ultimo elemento del scratchpad (que contiene el Final Answer o el error)
                return self.scratchpad[-1][2]
                
        return "Error: Se ha excedido el limite maximo de pasos de razonamiento (bucle infinito prevenido)."

    def _simulate_llm_response(self, task: str) -> str:
        """
        Simulador cognitivo que reproduce el comportamiento de un LLM paso a paso
        basandose en el estado actual de la bitacora de ejecucion (scratchpad).
        """
        task_lower = task.lower()
        num_steps = len(self.scratchpad)
        
        # Simulación de busqueda y resumen de NanoVectorDB
        if "nanovectordb" in task_lower or "vector" in task_lower:
            if num_steps == 0:
                return (
                    "Thought: Para resolver esta tarea, primero debo consultar la base de datos vectorial "
                    "para entender que es NanoVectorDB y que caracteristicas tecnicas posee.\n"
                    "Action: search_db('NanoVectorDB')"
                )
            elif num_steps == 1:
                return (
                    "Thought: He obtenido los datos tecnicos de NanoVectorDB (HNSW, similitud de coseno, in-memory). "
                    "Ahora necesito consolidar esta informacion redactando un informe estructurado.\n"
                    "Action: write_report('NanoVectorDB - Base de datos in-memory con indice HNSW')"
                )
            else:
                obs = self.scratchpad[-1][2]
                return (
                    f"Thought: Tengo el informe consolidado. Puedo responder al usuario.\n"
                    f"Final Answer: NanoVectorDB es una base de datos vectorial in-memory escrita en Python "
                    f"que implementa indexacion HNSW y similitud de coseno. {obs}"
                )
                
        # Simulación genérica
        if num_steps == 0:
            return (
                "Thought: Analizare la peticion general del usuario buscando coincidencias en herramientas registradas.\n"
                "Action: search_db('general query')"
            )
        else:
            return (
                "Thought: Tengo informacion suficiente para resolver la consulta general.\n"
                "Final Answer: Tarea resuelta exitosamente."
            )
