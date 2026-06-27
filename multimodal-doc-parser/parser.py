import os
from typing import List
from backend import ParserBackend

class MultimodalDocParser:
    """
    Clase principal que coordina el pipeline de parsing de documentos.
    
    Acepta archivos PDF e imágenes (PNG, JPG, JPEG) y utiliza un backend
    configurable para procesar el diseño visual y textual de cada página,
    devolviendo una representación unificada en Markdown.
    """
    
    SUPPORTED_IMAGES = (".png", ".jpg", ".jpeg")
    SUPPORTED_DOCS = (".pdf",)
    
    def __init__(self, backend: ParserBackend) -> None:
        """
        Args:
            backend: Proveedor de análisis que procesará cada página (Local o VLM).
        """
        self.backend = backend

    def _process_image(self, file_path: str) -> str:
        """
        Procesa un único archivo de imagen.
        """
        with open(file_path, "rb") as f:
            image_bytes = f.read()
            
        # Al ser una imagen directa, no tenemos texto digital de fallback
        # y la consideramos como la página 1.
        return self.backend.parse_page(image_bytes, "", 1)

    def _process_pdf(self, file_path: str) -> str:
        """
        Procesa un archivo PDF página por página.
        Utiliza PyMuPDF (fitz) para extraer texto nativo digital y
        renderizar cada página a imagen PNG en memoria de forma eficiente.
        """
        # Carga dinámica perezosa para evitar requerir fitz a menos que se use.
        try:
            import fitz
        except ImportError as e:
            raise ImportError(
                "No se pudo importar 'pymupdf' (fitz). Instálalo con: pip install pymupdf"
            ) from e
            
        doc = fitz.open(file_path)
        markdown_pages: List[str] = []
        
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # 1. Extracción de texto digital nativo (fallback)
                text_fallback = page.get_text()
                
                # 2. Renderización de la página a imagen PNG en memoria.
                # Aumentamos la resolución con una escala (zoom) de 2.0 (aproximadamente 150 DPI)
                # para asegurar una excelente calidad y legibilidad de texto pequeño.
                zoom = 2.0
                matrix = fitz.Matrix(zoom, zoom)
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                
                # Obtenemos los bytes de la imagen directamente como PNG
                image_bytes = pixmap.tobytes("png")
                
                # 3. Llamada al backend de parsing (1-indexed para el número de página)
                page_md = self.backend.parse_page(image_bytes, text_fallback, page_num + 1)
                markdown_pages.append(page_md)
                
        finally:
            doc.close()
            
        return "".join(markdown_pages)

    def parse(self, file_path: str) -> str:
        """
        Analiza el archivo especificado y devuelve su contenido en Markdown estructurado.
        
        Args:
            file_path: Ruta absoluta o relativa al archivo de entrada.
            
        Returns:
            Texto en formato Markdown.
            
        Raises:
            FileNotFoundError: Si el archivo no existe.
            ValueError: Si el formato del archivo no está soportado.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo especificado no existe: {file_path}")
            
        _, ext = os.path.splitext(file_path.lower())
        
        if ext in self.SUPPORTED_IMAGES:
            return self._process_image(file_path)
        elif ext in self.SUPPORTED_DOCS:
            return self._process_pdf(file_path)
        else:
            raise ValueError(
                f"Formato de archivo no soportado: '{ext}'. "
                f"Formatos válidos: {self.SUPPORTED_IMAGES + self.SUPPORTED_DOCS}"
            )
