import abc
import base64
import io
import os
from typing import Optional
from PIL import Image

class ParserBackend(abc.ABC):
    """
    Clase base abstracta para backends de parsing de documentos.
    
    Cada backend define su estrategia para transformar los datos visuales
    y digitales de una página de un documento en una salida de Markdown estructurado.
    """
    
    @abc.abstractmethod
    def parse_page(self, page_image_bytes: bytes, page_text_fallback: str, page_number: int) -> str:
        """
        Procesa una sola página y devuelve su representación en Markdown.
        
        Args:
            page_image_bytes: Representación en bytes (PNG) de la página renderizada.
            page_text_fallback: Texto nativo digital extraído directamente del archivo (si existe).
            page_number: Número de página secuencial (1-indexed).
            
        Returns:
            Contenido de la página formateado en Markdown.
        """
        pass


class LocalExtractBackend(ParserBackend):
    """
    Backend de extracción puramente local.
    
    Utiliza el texto digital nativo extraído del documento para construir
    un Markdown básico de forma gratuita, limpia y sin dependencias de red.
    """
    
    def parse_page(self, page_image_bytes: bytes, page_text_fallback: str, page_number: int) -> str:
        # Si no hay texto extraído nativamente (por ejemplo, en un PDF escaneado que solo contiene imágenes),
        # informamos que se requiere un backend de visión (VLM/OCR).
        if not page_text_fallback.strip():
            return f"\n\n--- PAGINA {page_number} ---\n\n*La pagina no contiene texto digital nativo. Se requiere un backend VLM para procesar esta pagina visualmente.*\n"
            
        # Limpiamos el texto y lo estructuramos en párrafos simples.
        paragraphs = page_text_fallback.split("\n\n")
        structured_paragraphs = []
        for p in paragraphs:
            cleaned = p.strip()
            if cleaned:
                # Si parece un encabezado corto, le damos formato de cabecera
                if len(cleaned) < 80 and not cleaned.endswith((".", ",", ";")):
                    structured_paragraphs.append(f"### {cleaned}")
                else:
                    structured_paragraphs.append(cleaned)
                    
        page_content = "\n\n".join(structured_paragraphs)
        return f"\n\n--- PAGINA {page_number} ---\n\n{page_content}\n"


class VLMParserBackend(ParserBackend):
    """
    Backend de análisis visual que utiliza Modelos de Lenguaje de Visión (VLM).
    
    Sostiene integración con la API de Google Gemini (por defecto, utilizando
    el modelo gemini-2.5-flash) y con OpenAI (utilizando gpt-4o-mini).
    
    Analiza la imagen de la página en busca de tablas, gráficos y distribuciones
    de texto complejas para renderizar un Markdown preciso de nivel de producción.
    """
    
    SYSTEM_PROMPT = (
        "Eres un analizador de documentos experto de nivel de producción. Tu tarea es transcribir "
        "la imagen de la página de este documento a formato Markdown estructurado (GitHub Flavored Markdown). "
        "Debes conservar todos los encabezados, párrafos y listas. "
        "Traduce cualquier tabla en la imagen a tablas de Markdown estructuradas de forma limpia. "
        "Si hay imágenes, diagramas, esquemas o gráficos, no los omitas; en su lugar, inserta "
        "un bloque explicativo descriptivo en cursiva que explique detalladamente lo que muestra el "
        "elemento visual (ej. *[Diagrama: Estructura del flujo de control de RAG que muestra...]*). "
        "Retorna únicamente el Markdown limpio, sin bloques de código con triple comilla (```markdown) "
        "ni explicaciones adicionales en el exterior."
    )
    
    def __init__(self, provider: str = "gemini", api_key: Optional[str] = None) -> None:
        """
        Args:
            provider: El VLM a utilizar ('gemini' o 'openai').
            api_key: Clave de API explícita. Si es None, se lee de variables de entorno.
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self._client = None
        
        if self.provider not in ("gemini", "openai"):
            raise ValueError("El proveedor debe ser 'gemini' o 'openai'.")

    def _get_client(self):
        """
        Inicialización y carga perezosa de los SDKs clientes correspondientes.
        """
        if self._client is not None:
            return self._client
            
        if self.provider == "gemini":
            key = self.api_key or os.environ.get("GEMINI_API_KEY")
            if not key:
                raise ValueError("No se ha configurado la variable de entorno GEMINI_API_KEY.")
            try:
                from google import genai
                # La API actual utiliza el cliente unificado de google-genai
                self._client = genai.Client(api_key=key)
            except ImportError as e:
                raise ImportError(
                    "No se pudo importar 'google-genai'. Instálalo con: pip install google-genai"
                ) from e
                
        elif self.provider == "openai":
            key = self.api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError("No se ha configurado la variable de entorno OPENAI_API_KEY.")
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=key)
            except ImportError as e:
                raise ImportError(
                    "No se pudo importar 'openai'. Instálalo con: pip install openai"
                ) from e
                
        return self._client

    def _parse_with_gemini(self, client, image_bytes: bytes) -> str:
        # Convertimos los bytes en un objeto de imagen PIL para el SDK de Google GenAI
        image = Image.open(io.BytesIO(image_bytes))
        
        # Invocamos el modelo gemini-2.5-flash usando el cliente unificado
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[image, self.SYSTEM_PROMPT]
        )
        return response.text if response.text else ""

    def _parse_with_openai(self, client, image_bytes: bytes) -> str:
        # OpenAI requiere la imagen codificada en base64
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.SYSTEM_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2000
        )
        return response.choices[0].message.content or ""

    def parse_page(self, page_image_bytes: bytes, page_text_fallback: str, page_number: int) -> str:
        try:
            client = self._get_client()
            
            if self.provider == "gemini":
                parsed_text = self._parse_with_gemini(client, page_image_bytes)
            else:
                parsed_text = self._parse_with_openai(client, page_image_bytes)
                
            # Limpiamos posibles bloques markdown envueltos con ```markdown o ``` al inicio/fin
            cleaned_text = parsed_text.strip()
            if cleaned_text.startswith("```markdown"):
                cleaned_text = cleaned_text[11:]
            elif cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:]
                
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
                
            return f"\n\n--- PAGINA {page_number} ---\n\n{cleaned_text.strip()}\n"
            
        except Exception as e:
            # Fallback seguro: reportamos el error e intentamos procesar de forma local
            error_msg = (
                f"\n\n--- PAGINA {page_number} ---\n\n"
                f"*Error al invocar la API del VLM ({e}). Cayendo en extraccion local de texto digital.*\n\n"
            )
            # Reutilizamos el backend local como alternativa
            local_fallback = LocalExtractBackend().parse_page(page_image_bytes, page_text_fallback, page_number)
            return error_msg + local_fallback
