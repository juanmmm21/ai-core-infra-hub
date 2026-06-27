import os
from backend import LocalExtractBackend, VLMParserBackend
from parser import MultimodalDocParser

def create_sample_pdf(file_path: str) -> None:
    """
    Crea programáticamente un archivo PDF digital de prueba utilizando PyMuPDF.
    Esto permite ejecutar la demo de inmediato sin requerir que el usuario provea un archivo externo.
    """
    try:
        import fitz
    except ImportError:
        print("Instale pymupdf para generar el archivo de prueba: pip install pymupdf")
        return
        
    doc = fitz.open()
    
    # Pagina 1: Introduccion
    page1 = doc.new_page()
    page1.insert_text((50, 80), "Reporte de Infraestructura de IA", fontsize=20, fontname="helv")
    page1.insert_text((50, 130), "Este documento sirve como prueba para verificar las capacidades del Multimodal Doc Parser.", fontsize=11, fontname="helv")
    page1.insert_text((50, 160), "El ecosistema consta de 21 submódulos independientes y reutilizables.", fontsize=11, fontname="helv")
    
    # Pagina 2: Datos
    page2 = doc.new_page()
    page2.insert_text((50, 80), "Analisis de Componentes", fontsize=20, fontname="helv")
    page2.insert_text((50, 130), "Seccion que contiene texto estructurado en multiples bloques y parrafos para validar", fontsize=11, fontname="helv")
    page2.insert_text((50, 150), "que el formateador local de parrafos agrupe los bloques correctamente.", fontsize=11, fontname="helv")
    
    doc.save(file_path)
    doc.close()
    print(f"✓ Archivo PDF de prueba '{file_path}' creado correctamente.")


def run_demo() -> None:
    sample_pdf_path = "test_sample.pdf"
    
    print("=== PASO 1: Creación del documento PDF de prueba ===")
    create_sample_pdf(sample_pdf_path)
    print("-" * 70)

    print("\n=== PASO 2: Procesamiento con Backend Local (Gratuito) ===")
    local_backend = LocalExtractBackend()
    local_parser = MultimodalDocParser(backend=local_backend)
    
    print(f"Analizando '{sample_pdf_path}' localmente...")
    local_markdown = local_parser.parse(sample_pdf_path)
    print("\n[Resultado del Markdown Local]:")
    print(local_markdown)
    print("-" * 70)

    print("\n=== PASO 3: Procesamiento con Backend de Visión (Gemini VLM) ===")
    # Verificamos si la API Key de Gemini está configurada
    api_key_configured = "GEMINI_API_KEY" in os.environ
    
    # Configuramos el backend de Gemini
    # Usamos una api_key dummy si no está configurada para que demuestre la caída segura de error (fallback)
    vlm_backend = VLMParserBackend(provider="gemini", api_key=None if api_key_configured else "DUMMY_KEY")
    vlm_parser = MultimodalDocParser(backend=vlm_backend)
    
    if not api_key_configured:
        print("Nota: GEMINI_API_KEY no está configurada en las variables de entorno.")
        print("El VLM simulará una caída de error segura y ejecutará el fallback al backend local.")
        
    print(f"Analizando '{sample_pdf_path}' con el Backend de Visión...")
    vlm_markdown = vlm_parser.parse(sample_pdf_path)
    
    print("\n[Resultado del Markdown de Visión / Fallback]:")
    print(vlm_markdown)
    print("-" * 70)

    print("\n=== PASO 4: Persistencia y Limpieza ===")
    output_path = "output_report.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(vlm_markdown)
    print(f"✓ Resultado de visión guardado en '{output_path}'.")
    
    # Limpiamos los archivos generados localmente
    if os.path.exists(sample_pdf_path):
        os.remove(sample_pdf_path)
        print(f"✓ Archivo temporal de prueba '{sample_pdf_path}' eliminado de forma limpia.")
        
    print("✓ Demostración completada.")
    print("-" * 70)

if __name__ == "__main__":
    run_demo()
