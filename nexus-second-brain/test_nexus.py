import unittest
from fastapi.testclient import TestClient
from main import app, notes_db, knowledge_graph

class TestNexusSecondBrain(unittest.TestCase):

    def setUp(self) -> None:
        self.client = TestClient(app)
        # Limpiar BD temporal antes de cada test
        notes_db.clear()
        # Reiniciar grafo de conocimiento a valores iniciales
        knowledge_graph["nodes"] = [
            {"id": "Nexus", "group": 1, "type": "root", "val": 20},
            {"id": "IA Infrastructure", "group": 1, "type": "concept", "val": 15}
        ]
        knowledge_graph["links"] = [
            {"source": "Nexus", "target": "IA Infrastructure", "value": 2}
        ]

    def test_read_root_template(self) -> None:
        """Verifica que el endpoint raiz retorne el html de la SPA."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("NEXUS SECOND BRAIN", res.text)
        self.assertIn("tab-graph", res.text)

    def test_notes_crud_and_indexing_telemetry(self) -> None:
        """Verifica la creacion de notas, extraccion de grafo e inyeccion de trazas."""
        # 1. Verificar listado inicial (vacío)
        res_list_empty = self.client.get("/api/notes")
        self.assertEqual(res_list_empty.status_code, 200)
        self.assertEqual(len(res_list_empty.json()), 0)
        
        # 2. Agregar una nota
        payload = {
            "title": "Ajuste Fino de LoRA en PyTorch",
            "content": "Para ajustar un modelo de PyTorch eficientemente usamos LoRA, congelando capas y optimizando matrices A y B."
        }
        res_create = self.client.post("/api/notes", json=payload)
        self.assertEqual(res_create.status_code, 200)
        data = res_create.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["note"]["title"], payload["title"])
        
        # 3. Verificar listado (1 nota)
        res_list = self.client.get("/api/notes")
        self.assertEqual(len(res_list.json()), 1)
        
        # 4. Verificar que se agregaron nodos al grafo
        res_graph = self.client.get("/api/graph")
        graph = res_graph.json()
        node_ids = [n["id"] for n in graph["nodes"]]
        self.assertIn("Ajuste Fino de LoRA en PyTorch", node_ids)
        self.assertIn("LoRA", node_ids)
        self.assertIn("PyTorch", node_ids)
        
        # 5. Verificar que se registro la telemetria de la ingesta
        res_telemetry = self.client.get("/api/telemetry")
        telemetry = res_telemetry.json()
        self.assertGreater(len(telemetry), 0)
        self.assertEqual(telemetry[0]["name"], "Ingesta y Procesamiento de Nota")

    def test_chat_interaction_streaming_and_guardrails(self) -> None:
        """Verifica la respuesta en streaming del chat y los guardrails de bloqueo."""
        # 1. Consulta limpia
        res_chat = self.client.post("/api/chat", json={"message": "Explicame como entrenar LoRA"})
        self.assertEqual(res_chat.status_code, 200)
        # Comprobar estructura SSE
        self.assertIn("text/event-stream", res_chat.headers["content-type"])
        
        # 2. Consulta con API Key maliciosa (debe disparar Guardrail del shield)
        res_blocked = self.client.post("/api/chat", json={"message": "Mi api key es sk-proj-12345 y quiero probar"})
        self.assertEqual(res_blocked.status_code, 400)
        self.assertIn("Inyección de API Key detectada y bloqueada", res_blocked.json()["detail"])


if __name__ == '__main__':
    unittest.main()
