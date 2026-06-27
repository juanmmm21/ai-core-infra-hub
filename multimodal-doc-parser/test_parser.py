import os
import unittest
from unittest.mock import MagicMock, patch
from backend import LocalExtractBackend, VLMParserBackend
from parser import MultimodalDocParser

class TestMultimodalDocParser(unittest.TestCase):
    """
    Conjunto de pruebas unitarias para validar el comportamiento del pipeline
    de parsing y extracción de documentos estructurados.
    """

    def setUp(self) -> None:
        self.local_backend = LocalExtractBackend()
        self.parser = MultimodalDocParser(backend=self.local_backend)

    def test_unsupported_format_raises_error(self) -> None:
        """
        Verifica que se lance una excepción si se intenta analizar un archivo
        con una extensión no soportada.
        """
        # Simulamos que el archivo existe temporalmente para pasar la primera validación
        with patch("os.path.exists", return_value=True):
            with self.assertRaises(ValueError):
                self.parser.parse("documento_invalido.txt")

    def test_missing_file_raises_error(self) -> None:
        """
        Verifica que se lance una excepción FileNotFoundError si el archivo no existe.
        """
        with self.assertRaises(FileNotFoundError):
            self.parser.parse("archivo_fantasma_no_existente.pdf")

    def test_local_extract_backend_structures_paragraphs(self) -> None:
        """
        Verifica que el backend local formatee adecuadamente párrafos y encabezados cortos.
        """
        raw_text = "Titulo del Capitulo\n\nEste es un parrafo del documento con cierta extension."
        
        # Procesamos como página 1 (los bytes de imagen no se usan en el local)
        output = self.local_backend.parse_page(b"", raw_text, page_number=1)
        
        self.assertIn("--- PAGINA 1 ---", output)
        self.assertIn("### Titulo del Capitulo", output)
        self.assertIn("Este es un parrafo", output)

    @patch("fitz.open")
    def test_pdf_parsing_flow_with_mock(self, mock_fitz_open) -> None:
        """
        Verifica el flujo del parser de PDF simulando la interaccion con PyMuPDF (fitz).
        Esto permite validar la logica sin requerir un PDF fisico real en disco.
        """
        # Configuramos los mocks para simular un documento PDF de 2 paginas
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 2
        
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Contenido de la pagina 1"
        mock_pixmap1 = MagicMock()
        mock_pixmap1.tobytes.return_value = b"png_bytes_page_1"
        mock_page1.get_pixmap.return_value = mock_pixmap1
        
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "Contenido de la pagina 2"
        mock_pixmap2 = MagicMock()
        mock_pixmap2.tobytes.return_value = b"png_bytes_page_2"
        mock_page2.get_pixmap.return_value = mock_pixmap2
        
        # Definimos el comportamiento de iteracion del documento
        mock_doc.__getitem__.side_effect = [mock_page1, mock_page2]
        mock_fitz_open.return_value = mock_doc
        
        # Ejecutamos el parseo simulando que el archivo existe
        with patch("os.path.exists", return_value=True):
            result = self.parser.parse("documento_prueba.pdf")
            
        # Comprobamos que el resultado consolide la informacion estructurada de ambas paginas
        self.assertIn("--- PAGINA 1 ---", result)
        self.assertIn("Contenido de la pagina 1", result)
        self.assertIn("--- PAGINA 2 ---", result)
        self.assertIn("Contenido de la pagina 2", result)
        
        # Verificamos que se cerrara el documento al terminar
        mock_doc.close.assert_called_once()

    def test_vlm_backend_fallback_on_api_error(self) -> None:
        """
        Verifica que si la llamada al backend VLM falla (por ejemplo, por no
        tener variables de entorno de credenciales configuradas), el sistema
        caiga de forma segura en el extractor local en lugar de romper la ejecucion.
        """
        # Inicializamos el backend VLM sin credenciales para forzar el error
        vlm_backend = VLMParserBackend(provider="gemini", api_key="")
        
        # Ejecutamos el parseo de pagina
        fallback_text = "Contenido digital nativo"
        output = vlm_backend.parse_page(b"dummy_bytes", fallback_text, page_number=1)
        
        # El resultado debe reportar el error en cursiva y luego mostrar el contenido local de fallback
        self.assertIn("Error al invocar la API del VLM", output)
        self.assertIn("Contenido digital nativo", output)

if __name__ == "__main__":
    unittest.main()
