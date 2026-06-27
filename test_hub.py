import unittest
import os
import sys
from fastapi.testclient import TestClient

# Setup path mapping to sibling projects
HUB_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(HUB_DIR)

if HUB_DIR not in sys.path:
    sys.path.append(HUB_DIR)

from main import app

class TestHubAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("AI Core Infra Hub", response.text)

    def test_bpe_tokenize(self):
        response = self.client.post("/api/bpe/tokenize", json={"text": "Hola mundo de tokens"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("tokens", data)
        self.assertIn("splits", data)

    def test_semantic_chunker(self):
        text = "Esta es la primera frase del texto. Esta es la segunda frase relacionada."
        response = self.client.post("/api/chunker/chunk", json={"text": text, "threshold": 1.2})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("chunks", data)

    def test_contrastive_similarity(self):
        response = self.client.post("/api/contrastive/similarity", json={"text1": "gatos", "text2": "felinos"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("similarity", data)
        self.assertIsInstance(data["similarity"], float)

    def test_guardrails_check(self):
        prompt = "Normal query with 555-0199 phone number."
        response = self.client.post("/api/guardrails/check", json={"prompt": prompt})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("input", data)
        self.assertTrue(data["input"]["safe"])

    def test_unified_pipeline(self):
        response = self.client.post("/api/pipeline/run", json={"prompt": "Test query HNSW index"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("response", data)
        self.assertIn("telemetry", data)

if __name__ == "__main__":
    unittest.main()
