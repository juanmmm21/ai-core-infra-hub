import unittest
from runtime import SecureToolRuntime, SecurityValidationError


class TestSecureToolRuntime(unittest.TestCase):
    """
    Suite de pruebas unitarias para validar las politicas de seguridad estatica AST
    y los limites de aislamiento de procesos (timeouts) en el runtime.
    """

    def setUp(self) -> None:
        self.runtime = SecureToolRuntime()

    def test_safe_code_execution(self) -> None:
        """
        Prueba que codigo matematico y logico estandar se ejecute exitosamente.
        """
        code = "a = 5\nb = 10\nprint(f'La suma es {a + b}')"
        success, stdout, stderr, err = self.runtime.run_code(code)
        
        self.assertTrue(success)
        self.assertEqual(stdout.strip(), "La suma es 15")
        self.assertEqual(stderr, "")
        self.assertEqual(err, "")

    def test_blocked_import_ast(self) -> None:
        """
        Prueba que la importacion de modulos peligrosos sea bloqueada estaticamente.
        """
        code_import = "import os\nos.system('ls')"
        success, stdout, stderr, err = self.runtime.run_code(code_import)
        
        self.assertFalse(success)
        self.assertIn("SecurityValidationError", err)
        self.assertIn("Importacion prohibida", err)

        code_import_from = "from sys import exit\nexit(0)"
        success2, _, _, err2 = self.runtime.run_code(code_import_from)
        self.assertFalse(success2)
        self.assertIn("Importacion prohibida", err2)

    def test_blocked_calls_ast(self) -> None:
        """
        Prueba que llamadas a funciones builtins peligrosas sean bloqueadas estaticamente.
        """
        codes = [
            "open('test.txt', 'r')",
            "eval('2 + 2')",
            "exec('a = 5')"
        ]
        
        for code in codes:
            success, _, _, err = self.runtime.run_code(code)
            self.assertFalse(success)
            self.assertIn("SecurityValidationError", err)
            self.assertIn("funcion prohibida", err)

    def test_timeout_protection(self) -> None:
        """
        Prueba que scripts con bucles infinitos sean terminados
        de forma segura al superar el limite de tiempo.
        """
        infinite_loop_code = "print('Iniciando loop...')\nwhile True:\n    pass"
        
        # Ejecutamos con un timeout estricto de 0.5 segundos para no demorar los tests
        success, stdout, stderr, err = self.runtime.run_code(infinite_loop_code, timeout_seconds=0.5)
        
        self.assertFalse(success)
        self.assertIn("TimeoutError", err)
        self.assertEqual(stdout, "")
        self.assertEqual(stderr, "")


if __name__ == "__main__":
    unittest.main()
