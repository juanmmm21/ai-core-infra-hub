import logging
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

from agent import AutonomousAgent

logger = logging.getLogger(__name__)


class TaskDelegation(BaseModel):
    task_id: int
    assigned_agent: str = Field(..., description="Nombre del agente al que se le delega la subtarea.")
    subtask_description: str = Field(..., description="Descripcion de la porcion de trabajo a realizar.")
    dependencies: List[int] = Field(default_factory=list, description="Lista de task_ids de los que depende.")


class AgentOrchestrator:
    """
    Orquestador de agentes autonomos (Multi-Agent Supervisor).
    
    Gestiona un equipo de agentes especializados, descompone una tarea global compleja
    en subtareas secuenciales o interdependientes y coordina la transferencia de contexto
    (message passing) y la consolidacion de resultados finales.
    """

    def __init__(self, name: str = "CentralSupervisor") -> None:
        self.name = name
        self.agents: Dict[str, AutonomousAgent] = {}
        self.delegation_plan: List[TaskDelegation] = []
        self.results_registry: Dict[int, str] = {}

    def register_agent(self, agent: AutonomousAgent) -> None:
        """Registra un agente especializado en el equipo de orquestacion."""
        self.agents[agent.name] = agent
        logger.info(f"Agente [{agent.name}] registrado en el orquestador '{self.name}'.")

    def plan_delegation(self, global_task: str) -> List[TaskDelegation]:
        """
        Descompone de forma analitica la tarea global en subtareas logicas
        asignando cada una al agente idoneo en el pool.
        """
        self.delegation_plan.clear()
        self.results_registry.clear()
        
        task_lower = global_task.lower()
        
        # Simulacion de planificacion dinamica. En produccion, esto se realiza mediante un
        # agente planificador LLM (Planner Agent) que genera un JSON con la estructura de tareas.
        if "nanovectordb" in task_lower or "informe" in task_lower:
            # Plan estructurado: 1. Investigar (Researcher), 2. Redactar (Writer)
            self.delegation_plan.append(
                TaskDelegation(
                    task_id=1,
                    assigned_agent="ResearcherAgent",
                    subtask_description="Buscar especificaciones tecnicas sobre NanoVectorDB en la base de datos.",
                    dependencies=[]
                )
            )
            self.delegation_plan.append(
                TaskDelegation(
                    task_id=2,
                    assigned_agent="WriterAgent",
                    subtask_description="Redactar un resumen ejecutivo y consolidado basado en los hallazgos de la investigacion.",
                    dependencies=[1]
                )
            )
        else:
            # Plan generico de paso unico
            self.delegation_plan.append(
                TaskDelegation(
                    task_id=1,
                    assigned_agent="ResearcherAgent",
                    subtask_description=f"Procesar la consulta general: {global_task}",
                    dependencies=[]
                )
            )
            
        return self.delegation_plan

    def execute(self, global_task: str) -> str:
        """
        Ejecuta el plan de delegacion secuencialmente respetando las dependencias de datos
        y transmitiendo el output del agente anterior como contexto para el siguiente.
        """
        plan = self.plan_delegation(global_task)
        logger.info(f"Iniciando ejecucion de plan de orquestacion con {len(plan)} pasos...")
        
        for step in plan:
            agent_name = step.assigned_agent
            if agent_name not in self.agents:
                raise ValueError(f"Error de orquestacion: El agente asignado '{agent_name}' no esta registrado.")
                
            agent = self.agents[agent_name]
            
            # Recolectamos resultados de dependencias previas como contexto
            context_pieces = []
            for dep_id in step.dependencies:
                if dep_id in self.results_registry:
                    context_pieces.append(f"[Resultado Tarea {dep_id}]: {self.results_registry[dep_id]}")
                    
            context_str = "\n".join(context_pieces)
            
            # Construimos la peticion enriquecida para el agente asignado
            current_task = step.subtask_description
            if context_str:
                current_task += f"\n\nContexto de apoyo previo:\n{context_str}"
                
            # Ejecutamos el agente de forma autonoma
            logger.info(f"Orquestador delegando subtarea {step.task_id} a [{agent_name}]...")
            result = agent.run(current_task)
            
            # Registramos el resultado de este paso
            self.results_registry[step.task_id] = result
            
        # El resultado final del orquestador es la resolucion del ultimo paso del plan
        final_task_id = plan[-1].task_id
        return self.results_registry[final_task_id]
