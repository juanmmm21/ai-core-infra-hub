import unittest
import time
from memory import AgenticMemory


class TestAgenticMemory(unittest.TestCase):
    """
    Suite de pruebas unitarias para certificar la memoria de corto plazo (sintesis)
    y la memoria de largo plazo (recuperacion vectorial con decaimiento temporal).
    """

    def setUp(self) -> None:
        import os
        if os.path.exists("agent_long_term_memory.json"):
            try:
                os.remove("agent_long_term_memory.json")
            except OSError:
                pass
        # Configuramos una memoria con decaimiento rapido (factor 0.5) para pruebas controladas
        self.memory = AgenticMemory(short_term_max_turns=2, decay_factor=0.5, vector_dim=16)

    def test_short_term_summarization(self) -> None:
        """
        Verifica que al sobrepasar el maximo de turnos en corto plazo,
        se libere el historial y se genere un resumen acumulado.
        """
        # Añadimos 5 interacciones de usuario y asistente (10 turnos)
        for i in range(5):
            self.memory.add_interaction("user", f"Mensaje del usuario {i} con datos importantes")
            self.memory.add_interaction("assistant", f"Respuesta {i} del asistente")
            
        # El buffer de corto plazo debe haberse recortado a las ultimas 2 interacciones (4 turnos)
        self.assertTrue(len(self.memory.short_term) <= 4)
        # El resumen de la conversacion debe haberse compilado con informacion previa
        self.assertTrue(len(self.memory.conversation_summary) > 0)
        self.assertIn("Mensaje del usuario 0", self.memory.conversation_summary)

    def test_long_term_recall_and_decay(self) -> None:
        """
        Verifica que los hechos de largo plazo se recuperen y decaigan exponencialmente
        con el paso del tiempo simulado.
        """
        t0 = time.time()
        
        # Guardamos un hecho importante en t0
        self.memory.save_fact("Al usuario le encanta programar en Python", importance=8)
        # Guardamos otro hecho de diferente tematica
        self.memory.save_fact("El usuario viaja a Paris en avion", importance=5)
        
        # Consulta inmediata (t = t0)
        results_t0 = self.memory.recall("lenguaje de programacion Python", top_k=1, current_time=t0)
        self.assertEqual(len(results_t0), 1)
        self.assertEqual(results_t0[0]["fact"], "Al usuario le encanta programar en Python")
        score_t0 = results_t0[0]["recall_score"]
        
        # Consulta tardia (t = t0 + 10 segundos de retraso simulado)
        t10 = t0 + 10.0
        results_t10 = self.memory.recall("lenguaje de programacion Python", top_k=1, current_time=t10)
        score_t10 = results_t10[0]["recall_score"]
        
        # El score en t10 debe ser menor debido al decaimiento temporal exponencial de olvido
        self.assertTrue(score_t10 < score_t0)
        
        # Verificamos la tasa de decaimiento teorica: e^(-0.5 * 10) = e^(-5) ≈ 0.0067
        # Por lo tanto score_t10 deberia ser aprox score_t0 * 0.0067
        ratio = score_t10 / score_t0
        self.assertAlmostEqual(ratio, 0.0067379, places=3)


if __name__ == "__main__":
    unittest.main()
