import unittest
import os
import tempfile
import shutil
import time
from tracer import tracer, trace_span

class TestLlmObservabilityTracer(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        tracer.clear()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)
        tracer.clear()

    def test_hierarchical_nesting_and_durations(self) -> None:
        """Verifica que los spans se aniden jerárquicamente en árbol y midan tiempos."""
        
        @trace_span("funcion_padre")
        def parent_func():
            time.sleep(0.01)
            child_func()
            
        @trace_span("funcion_hijo")
        def child_func():
            time.sleep(0.005)
            
        parent_func()
        
        tree = tracer.get_trace_tree()
        self.assertEqual(len(tree), 1)
        
        root = tree[0]
        self.assertEqual(root["name"], "funcion_padre")
        self.assertGreater(root["end_time"], root["start_time"])
        
        self.assertEqual(len(root["children"]), 1)
        child = root["children"][0]
        self.assertEqual(child["name"], "funcion_hijo")
        self.assertEqual(child["parent_id"], root["id"])

    def test_metadata_and_error_capture(self) -> None:
        """Verifica que se capturen correctamente las entradas, salidas y excepciones."""
        
        @trace_span("operacion_matematica")
        def divide(a, b):
            return a / b
            
        # Invocación exitosa
        res = divide(10, 2)
        self.assertEqual(res, 5.0)
        
        tree = tracer.get_trace_tree()
        self.assertEqual(len(tree), 1)
        self.assertEqual(tree[0]["inputs"]["arg_0"], "10")
        self.assertEqual(tree[0]["inputs"]["arg_1"], "2")
        self.assertEqual(tree[0]["outputs"]["result"], 5.0)
        self.assertIsNone(tree[0]["error"])
        
        # Invocación errónea
        tracer.clear()
        with self.assertRaises(ZeroDivisionError):
            divide(10, 0)
            
        tree_err = tracer.get_trace_tree()
        self.assertEqual(len(tree_err), 1)
        self.assertEqual(tree_err[0]["inputs"]["arg_0"], "10")
        self.assertEqual(tree_err[0]["inputs"]["arg_1"], "0")
        # El resultado de salida debe estar vacío o no contener success
        self.assertEqual(tree_err[0]["outputs"], {})
        # Debe contener la firma del error
        self.assertIsNotNone(tree_err[0]["error"])
        self.assertIn("ZeroDivisionError", tree_err[0]["error"])

    def test_token_and_cost_aggregation(self) -> None:
        """Verifica que se puedan inyectar metadatos de tokens y costes en spans."""
        
        tracer.start_span("llm_call", inputs={"prompt": "Hola"})
        # Simulamos llamada LLM asignando tokens
        span = tracer.span_map[list(tracer.span_map.keys())[0]]
        span.metadata["input_tokens"] = 10
        span.metadata["output_tokens"] = 20
        span.metadata["cost"] = 0.000045
        
        tracer.end_span(outputs={"response": "Mundo"})
        
        tree = tracer.get_trace_tree()
        self.assertEqual(tree[0]["metadata"]["input_tokens"], 10)
        self.assertEqual(tree[0]["metadata"]["output_tokens"], 20)
        self.assertEqual(tree[0]["metadata"]["cost"], 0.000045)

    def test_html_and_ascii_exports(self) -> None:
        """Prueba las utilidades de exportacion de diagramas de observabilidad."""
        tracer.start_span("root")
        tracer.start_span("sub_step")
        tracer.end_span()
        tracer.end_span()
        
        # ASCII export
        ascii_graph = tracer.generate_ascii_flamegraph()
        self.assertIn("- root", ascii_graph)
        self.assertIn("  - sub_step", ascii_graph)
        
        # HTML export
        html_path = os.path.join(self.temp_dir, "flamegraph.html")
        tracer.generate_html_flamegraph(html_path)
        self.assertTrue(os.path.exists(html_path))
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("Monitoreo End-to-End", content)
            document_js_data = 'const traceData = ['
            self.assertIn("traceData", content)


if __name__ == '__main__':
    unittest.main()
