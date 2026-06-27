import unittest
from router import SemanticModelRouter


class TestSemanticModelRouter(unittest.TestCase):
    """
    Suite de pruebas unitarias para validar las decisiones del enrutador de modelos
    y sus calculos de complejidad y coste.
    """

    def setUp(self) -> None:
        self.router = SemanticModelRouter()

    def test_low_complexity_routing(self) -> None:
        """
        Verifica que mensajes cortos o basicos vayan al modelo mas economico.
        """
        prompts = [
            "Hola buenos dias",
            "OK gracias",
            "¿Como te llamas?",
            "   "  # Prompts vacios o casi vacios
        ]
        
        for p in prompts:
            decision = self.router.route(p)
            self.assertEqual(decision.selected_model, "gpt-4o-mini")
            self.assertEqual(decision.complexity_category, "low")
            self.assertEqual(decision.routing_path, "regex_rules")

    def test_high_complexity_coding_routing(self) -> None:
        """
        Verifica que los prompts que contienen codigo o patrones de desarrollo
        se enruten al modelo mas capaz.
        """
        prompts = [
            "escribe una funcion def calculate_pi(n: int) -> float para aproximar pi",
            "como importar numpy y hacer un struct impl en rust?",
            "Optimiza este codigo: for i in range(len(l)): print(l[i])",
            "Escribe una consulta SQL con inner join para traer los usuarios y compras"
        ]
        
        for p in prompts:
            decision = self.router.route(p)
            self.assertEqual(decision.selected_model, "claude-3-5-sonnet")
            self.assertEqual(decision.complexity_category, "high")

    def test_medium_complexity_reasoning_routing(self) -> None:
        """
        Verifica que los prompts de complejidad intermedia vayan a gpt-4o.
        """
        prompts = [
            "Realiza un analisis exhaustivo de las ventajas y desventajas de la energia solar en el sur de España",
            "Explica paso a paso como funciona el algoritmo de Dijkstra para grafos sin meter codigo",
            "Deduce cual es el proximo elemento de la serie y justifica tu respuesta"
        ]
        
        for p in prompts:
            decision = self.router.route(p)
            self.assertEqual(decision.selected_model, "gpt-4o")
            self.assertEqual(decision.complexity_category, "medium")

    def test_cost_calculation(self) -> None:
        """
        Verifica que el coste estimado se calcule de forma proporcional (75% input, 25% output).
        """
        decision = self.router.route("hola")
        profile = self.router.model_map["gpt-4o-mini"]
        expected_cost = (profile.cost_per_1k_input * 0.75) + (profile.cost_per_1k_output * 0.25)
        self.assertAlmostEqual(decision.estimated_cost_per_1k_tokens, expected_cost, places=7)


if __name__ == "__main__":
    unittest.main()
