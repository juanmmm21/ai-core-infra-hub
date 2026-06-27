# Multimodal Document Parser

Este subproyecto forma parte de la infraestructura modular de Inteligencia Artificial ai-core-infra. Implementa un pipeline de parsing y extraccion de documentos (Multimodal Document Parser) capaz de transformar archivos PDF e imagenes complejas con disenos variables en documentos estructurados en formato Markdown (GitHub Flavored Markdown).

Para este sistema se utiliza PyMuPDF (fitz) para renderizar y manipular las paginas directamente en memoria, permitiendo un pipeline portable de alto rendimiento sin requerir dependencias complejas de sistema como poppler.

## Arquitectura de Ingesta y Fundamentos de Renderizado

El pipeline consta de tres fases fundamentales:

### 1. Lectura y Renderizacion en Alta Resolucion
El parser recibe el archivo de entrada. Si es una imagen (PNG, JPG, JPEG), se lee directamente en bytes. Si es un archivo PDF, se abre y cada pagina se renderiza como una imagen PNG de alta resolucion aplicando una matriz de transformacion de escala:

$$\text{Matrix}(s_x, s_y) = \begin{pmatrix} s_x & 0 & 0 \\ 0 & s_y & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

Con $s_x = s_y = 2.0$, lo cual duplica las dimensiones de la caja de limites (bounding box) de la pagina (escalando la resolucion estandar de 72 DPI a 144 DPI). Esto garantiza que el texto pequeno y los bordes de las tablas sean legibles para los modelos de vision.

### 2. Abstraccion de Backends y Estrategias de Extraccion
Se define una interfaz abstracta `ParserBackend` que permite alternar la estrategia de procesamiento:

*   **LocalExtractBackend (Local):** Extractor gratuito que obtiene el texto digital nativo del PDF utilizando PyMuPDF y lo organiza en parrafos y cabeceras estructuradas mediante heuristica local de tamano de fuente y espaciado vertical.
*   **VLMParserBackend (API de Vision):** Envias la pagina renderizada (en bytes PNG) a un Modelo de Lenguaje de Vision (VLM) como Google Gemini (`gemini-2.5-flash`) o GPT-4o de OpenAI. El modelo recibe un prompt de sistema que le instruye transcribir de forma exacta la estructura del documento en formato Markdown GFM:
    *   Preservar tablas medibles en tablas Markdown (`| col1 | col2 |`).
    *   Preservar jerarquia de cabeceras (`#`, `##`, `###`).
    *   Reemplazar figuras e imagenes por anotaciones descriptivas explicitas: `* [Figura: descripcion detallada] *`.

### 3. Tolerancia a Fallos y Fallback Local Resiliente
Si la llamada de red al API del VLM falla (debido a limites de tasa, error de red o credenciales ausentes), el pipeline captura la excepcion, reporta una advertencia y realiza un fallback automatico y seguro al `LocalExtractBackend`, garantizando la continuidad de la ejecucion:

```python
try:
    markdown_content = vlm_backend.parse_page(page_image)
except Exception as e:
    logger.warning(f"Fallo en VLM. Activando fallback local. Motivo: {e}")
    markdown_content = local_backend.parse_page(page)
```

## Requisitos de Instalacion

*   Python 3.10 o superior
*   PyMuPDF (fitz)
*   Pillow
*   Google-Genai
*   OpenAI

Para instalar las dependencias locales, ejecute:
```bash
pip install -r requirements.txt
```

## Configuración de Entorno (Opcional)

Para habilitar el VLM real:
```bash
export GEMINI_API_KEY="tu-clave-de-api-de-google-genai"
# O bien
export OPENAI_API_KEY="tu-clave-de-api-de-openai"
```

## Guia de Ejecucion y Verificacion

### 1. Ejecutar Pruebas Unitarias
Las pruebas simulan paginas PDF ficticias mediante `unittest.mock` para verificar la resiliencia y el fallback:
```bash
python3 -m unittest test_parser.py
```

### 2. Ejecutar Demostracion
```bash
python3 example.py
```
El script generara un documento PDF de prueba interactivo, lo procesara con ambos backends y escribira la salida unificada en `output_report.md`.

## Conectividad en el Ecosistema ai-core-infra

El modulo `multimodal-doc-parser` es el punto de entrada de documentos complejos:
*   Sus outputs en Markdown estructurado alimentan al [semantic-chunking-engine](https://github.com/juanmmm21/semantic-chunking-engine).
*   Se interconecta con [bpe-tokenizer-from-scratch](https://github.com/juanmmm21/bpe-tokenizer-from-scratch) para verificar que las paginas extraidas no superen limites fisicos.
*   Los parrafos y tablas estructuradas son indexados semanticamente en [nano-vector-db](https://github.com/juanmmm21/nano-vector-db) para habilitar busquedas hibridas en [nexus-second-brain](https://github.com/juanmmm21/nexus-second-brain).
