import unittest
from generator import SyntheticDataGenerator, InstructionSample, DPOSample


class TestSyntheticDataGenerator(unittest.TestCase):
    """
    Suite de pruebas unitarias para certificar la generacion de datos sinteticos,
    el calculo de overlap de n-gramas y los filtros de desduplicacion.
    """

    def setUp(self) -> None:
        # Instanciamos el generador con un threshold estricto para forzar desduplicacion en test
        self.generator = SyntheticDataGenerator(min_length=10, dedup_threshold=0.40)

    def test_trigram_overlap(self) -> None:
        """
        Verifica el calculo de similitud de Jaccard basada en tri-gramas.
        """
        text1 = "Escribe una funcion en Python"
        text2 = "Escribe una funcion en Rust"
        
        overlap = self.generator.calculate_overlap(text1, text2)
        # Debe haber solapamiento alto debido a "Escribe una funcion en "
        self.assertTrue(overlap > 0.40)
        
        text3 = "Tortilla de patatas receta"
        overlap_low = self.generator.calculate_overlap(text1, text3)
        # Solapamiento nulo o muy bajo
        self.assertEqual(overlap_low, 0.0)

    def test_instruction_generation_format(self) -> None:
        """
        Prueba que el dataset de instrucciones se genere con el tipado e integridad correctos.
        """
        topics = ["RAG", "embeddings"]
        dataset = self.generator.generate_instruction_dataset(topics, count_per_topic=3)
        
        # Deben haberse generado 6 muestras (3 por topico)
        self.assertEqual(len(dataset), 6)
        
        for sample in dataset:
            self.assertIsInstance(sample, InstructionSample)
            self.assertTrue(len(sample.instruction) >= 10)
            self.assertTrue(len(sample.output) >= 20)
            # Validar que los prompts no tengan duplicados exactos
            self.assertTrue(any(t in sample.instruction.lower() for t in ["rag", "embeddings", "recuperacion", "vector", "dimension"]))

    def test_dpo_generation_format(self) -> None:
        """
        Prueba que el dataset DPO contenga los campos de prompt, chosen y rejected.
        """
        topics = ["db"]
        dataset = self.generator.generate_dpo_dataset(topics, count_per_topic=2)
        
        self.assertEqual(len(dataset), 2)
        for sample in dataset:
            self.assertIsInstance(sample, DPOSample)
            self.assertTrue(len(sample.prompt) > 0)
            self.assertTrue(len(sample.chosen) > 0)
            self.assertTrue(len(sample.rejected) > 0)
            self.assertNotEqual(sample.chosen, sample.rejected)


if __name__ == "__main__":
    unittest.main()
