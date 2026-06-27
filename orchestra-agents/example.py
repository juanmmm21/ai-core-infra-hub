from agent import AutonomousAgent
from orchestrator import AgentOrchestrator


def run_demo() -> None:
    print("=" * 70)
    print("      Demostracion de Orchestra Agents (Multi-Agent System)      ")
    print("=" * 70)
    
    # 1. Definir herramientas simuladas de infraestructura
    def search_vector_db(query: str) -> str:
        print(f"  [TOOL EXECUTION] search_db('{query}')...")
        if "nanovectordb" in query.lower() or "vector" in query.lower():
            return (
                "HALLAZGOS: NanoVectorDB es una base de datos vectorial in-memory. "
                "Utiliza indice HNSW, soporta distancias L2 y Cosina, y tiene filtrado "
                "de metadatos estilo MongoDB."
            )
        return "No se encontraron registros sobre esta consulta."

    def write_markdown_report(content: str) -> str:
        print(f"  [TOOL EXECUTION] write_report('...')...")
        return (
            f"### INFORME TECNICO OFICIAL\n"
            f"Consolidado de infraestructura: {content}\n"
            f"Estado del modulo: LISTO PARA PRODUCCION."
        )

    tools_catalog = {
        "search_db": search_vector_db,
        "write_report": write_markdown_report
    }

    # 2. Inicializar los agentes autonomos especializados
    researcher = AutonomousAgent(
        name="ResearcherAgent",
        role="Investigador de Sistemas de IA",
        system_prompt=(
            "Eres un agente investigador. Tu objetivo es buscar informacion tecnica y precisa. "
            "Usa la herramienta search_db para buscar conceptos en la base de datos."
        ),
        tools=tools_catalog,
        use_mock_llm=True
    )

    writer = AutonomousAgent(
        name="WriterAgent",
        role="Redactor de Documentacion Tecnica",
        system_prompt=(
            "Eres un agente redactor. Tomas hallazgos de investigacion y redactas un "
            "informe en Markdown usando la herramienta write_report."
        ),
        tools=tools_catalog,
        use_mock_llm=True
    )

    # 3. Inicializar el Orquestador
    orchestrator = AgentOrchestrator(name="DevOpsSupervisor")
    orchestrator.register_agent(researcher)
    orchestrator.register_agent(writer)

    # 4. Lanzar la meta-tarea
    task = "Investigar sobre NanoVectorDB y escribir un informe de arquitectura."
    print(f"\nMeta-Tarea: '{task}'\n")
    
    final_output = orchestrator.execute(task)
    
    print("\n" + "=" * 70)
    print("      RESULTADO FINAL CONSOLIDADO DEL SISTEMA MULTI-AGENTE      ")
    print("=" * 70)
    print(final_output)


if __name__ == "__main__":
    run_demo()
