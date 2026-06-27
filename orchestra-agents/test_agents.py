import unittest
from agent import AutonomousAgent
from orchestrator import AgentOrchestrator, TaskDelegation


class TestOrchestraAgents(unittest.TestCase):
    """
    Suite de pruebas unitarias para certificar el parseo ReAct de los agentes,
    la ejecucion de herramientas y el flujo de delegacion del orquestador.
    """

    def setUp(self) -> None:
        # Herramientas de prueba
        self.mock_tools = {
            "search_db": lambda q: f"Datos recuperados para la query '{q}': NanoVectorDB utiliza HNSW.",
            "write_report": lambda content: f"Informe guardado exitosamente con contenido: {content}"
        }
        
        self.agent = AutonomousAgent(
            name="TestAgent",
            role="Tester",
            system_prompt="Escribe en formato ReAct.",
            tools=self.mock_tools,
            use_mock_llm=True
        )

    def test_react_parsing(self) -> None:
        """
        Verifica el parseo correcto de la salida estructurada de los agentes.
        """
        output_action = "Thought: Necesito consultar la BD.\nAction: search_db('HNSW Index')"
        thought, action, arg, final = self.agent._parse_react_output(output_action)
        self.assertEqual(thought, "Necesito consultar la BD.")
        self.assertEqual(action, "search_db")
        self.assertEqual(arg, "HNSW Index")
        self.assertIsNone(final)
        
        output_final = "Thought: Ya tengo todo.\nFinal Answer: El resultado es 42."
        thought, action, arg, final = self.agent._parse_react_output(output_final)
        self.assertEqual(thought, "Ya tengo todo.")
        self.assertIsNone(action)
        self.assertIsNone(arg)
        self.assertEqual(final, "El resultado es 42.")

    def test_react_loop_execution(self) -> None:
        """
        Prueba la ejecucion completa del bucle ReAct del agente y la llamada a herramientas.
        """
        # Ejecuta la tarea simulada de NanoVectorDB
        result = self.agent.run("investigar NanoVectorDB")
        
        # Debio pasar por pasos: search_db y write_report
        self.assertEqual(len(self.agent.scratchpad), 3)
        self.assertEqual(self.agent.scratchpad[0][1], "search_db(NanoVectorDB)")
        self.assertEqual(self.agent.scratchpad[1][1], "write_report(NanoVectorDB - Base de datos in-memory con indice HNSW)")
        self.assertIn("NanoVectorDB es una base de datos vectorial", result)

    def test_orchestrator_delegation_flow(self) -> None:
        """
        Prueba que el orquestador cree un plan y delegue correctamente
        a investigadores y redactores.
        """
        orchestrator = AgentOrchestrator()
        
        # Creamos dos agentes y los registramos
        researcher = AutonomousAgent(
            name="ResearcherAgent",
            role="Investigador",
            system_prompt="Busca informacion.",
            tools=self.mock_tools,
            use_mock_llm=True
        )
        writer = AutonomousAgent(
            name="WriterAgent",
            role="Redactor",
            system_prompt="Escribe informes.",
            tools=self.mock_tools,
            use_mock_llm=True
        )
        
        orchestrator.register_agent(researcher)
        orchestrator.register_agent(writer)
        
        # Ejecutamos tarea global
        final_result = orchestrator.execute("Hacer un informe sobre NanoVectorDB")
        
        # Verificamos que se hayan ejecutado ambas tareas en orden
        self.assertIn(1, orchestrator.results_registry)
        self.assertIn(2, orchestrator.results_registry)
        self.assertIn("NanoVectorDB es una base de datos vectorial", final_result)


if __name__ == "__main__":
    unittest.main()
