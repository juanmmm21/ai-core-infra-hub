import os
import json
import time
import uuid
import logging
import functools
from contextvars import ContextVar
from typing import Dict, List, Any, Tuple, Optional, Callable
from pydantic import BaseModel, Field

# Logger para auditoría telemetral
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpanModel(BaseModel):
    """Representación de datos estructurados para un Span de telemetría."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    name: str
    start_time: float
    end_time: Optional[float] = None
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    children: List["SpanModel"] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        """Devuelve la latencia de ejecución en segundos."""
        if self.end_time is None:
            return 0.0
        return self.end_time - self.start_time


# ContextVar para rastrear el Span activo de forma segura en entornos async e hilos
_active_span_var: ContextVar[Optional[SpanModel]] = ContextVar("active_span", default=None)


class GlobalTracer:
    """
    Motor de Observabilidad central (Tracer).
    
    Gestiona el ciclo de vida de los Spans en una estructura de árbol jerárquica,
    acumula estadísticas agregadas (tokens, coste) y exporta trazas en ASCII e HTML.
    """

    def __init__(self) -> None:
        self.root_spans: List[SpanModel] = []
        # Mapa plano para accesos rápidos por ID
        self.span_map: Dict[str, SpanModel] = {}

    def start_span(
        self,
        name: str,
        inputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SpanModel:
        """Inicializa un nuevo span jerárquico bajo el span activo actual."""
        parent = _active_span_var.get()
        
        span = SpanModel(
            name=name,
            parent_id=parent.id if parent else None,
            start_time=time.time(),
            inputs=inputs or {},
            metadata=metadata or {}
        )
        
        self.span_map[span.id] = span
        
        if parent:
            parent.children.append(span)
        else:
            self.root_spans.append(span)
            
        _active_span_var.set(span)
        return span

    def end_span(
        self,
        outputs: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None,
        metadata_update: Optional[Dict[str, Any]] = None
    ) -> None:
        """Finaliza el span activo actual, registrando salidas, errores y duraciones."""
        span = _active_span_var.get()
        if not span:
            logger.warning("Intento de finalizar un span sin ningún span activo en el contexto.")
            return
            
        span.end_time = time.time()
        if outputs:
            span.outputs.update(outputs)
        if error:
            span.error = f"{type(error).__name__}: {str(error)}"
        if metadata_update:
            span.metadata.update(metadata_update)
            
        # Retornamos al span padre en el contexto actual
        if span.parent_id:
            parent = self.span_map.get(span.parent_id)
            _active_span_var.set(parent)
        else:
            _active_span_var.set(None)

    def clear(self) -> None:
        """Reinicia el historial de trazas acumulado."""
        self.root_spans = []
        self.span_map = {}
        _active_span_var.set(None)

    def get_trace_tree(self) -> List[Dict[str, Any]]:
        """Devuelve el árbol completo de trazas serializado en diccionarios."""
        return [span.model_dump() for span in self.root_spans]

    def generate_ascii_flamegraph(self) -> str:
        """
        Genera un diagrama de barras ASCII (Flame Graph de texto) detallando
        la jerarquía de ejecución, la latencia y los tokens consumidos.
        """
        lines = []
        
        def _render_span(span: SpanModel, depth: int = 0) -> None:
            indent = "  " * depth
            duration_ms = span.duration * 1000
            
            # Obtener tokens si están registrados en el metadata
            input_tokens = span.metadata.get("input_tokens", 0)
            output_tokens = span.metadata.get("output_tokens", 0)
            token_str = ""
            if input_tokens or output_tokens:
                token_str = f" | [In:{input_tokens} Out:{output_tokens} tks]"
                
            cost = span.metadata.get("cost", 0.0)
            cost_str = f" | [${cost:.6f}]" if cost > 0.0 else ""
            
            status = " [OK]"
            if span.error:
                status = " [ERROR]"
                
            lines.append(
                f"{indent}- {span.name}{status}: {duration_ms:.2f}ms{token_str}{cost_str}"
            )
            
            for child in span.children:
                _render_span(child, depth + 1)
                
        for root in self.root_spans:
            _render_span(root)
            
        return "\n".join(lines)

    def generate_html_flamegraph(self, filepath: str) -> None:
        """
        Exporta las trazas recolectadas a una aplicación web interactiva local de observabilidad
        con una interfaz moderna en Dark Mode y representación gráfica de la línea temporal.
        """
        # Convertimos la estructura de datos a JSON seguro para embeber en JS
        trace_json = json.dumps(self.get_trace_tree(), indent=2, ensure_ascii=False)
        
        html_template = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Visualizador de Trazas de IA (Observability)</title>
    <style>
        body {{
            background-color: #0b0f19;
            color: #e2e8f0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #1e1b4b, #311042);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #4338ca;
            margin-bottom: 25px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }}
        h1 {{
            margin: 0;
            font-size: 24px;
            color: #a5b4fc;
        }}
        .summary-stats {{
            display: flex;
            gap: 20px;
            margin-top: 15px;
        }}
        .stat-card {{
            background: rgba(255,255,255,0.05);
            padding: 10px 20px;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .stat-val {{
            font-size: 18px;
            font-weight: bold;
            color: #f472b6;
        }}
        .flame-container {{
            background: #111827;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #374151;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }}
        .span-row {{
            position: relative;
            height: 35px;
            margin-bottom: 6px;
            background: rgba(255,255,255,0.02);
            border-radius: 4px;
        }}
        .span-bar {{
            position: absolute;
            height: 100%;
            border-radius: 4px;
            background: linear-gradient(90deg, #6366f1, #a855f7);
            border: 1px solid #818cf8;
            box-sizing: border-box;
            cursor: pointer;
            transition: filter 0.2s;
            display: flex;
            align-items: center;
            padding: 0 10px;
            overflow: hidden;
            white-space: nowrap;
            text-overflow: ellipsis;
            font-size: 12px;
            font-weight: 500;
            color: #ffffff;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }}
        .span-bar:hover {{
            filter: brightness(1.2);
        }}
        .span-bar.error-bar {{
            background: linear-gradient(90deg, #ef4444, #f97316);
            border-color: #f87171;
        }}
        .details-panel {{
            margin-top: 25px;
            background: #1f2937;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #4b5563;
            display: none;
        }}
        .details-panel h3 {{
            margin-top: 0;
            color: #a5b4fc;
            border-bottom: 1px solid #374151;
            padding-bottom: 8px;
        }}
        pre {{
            background: #0f172a;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            border: 1px solid #1e293b;
            color: #38bdf8;
            font-family: 'Courier New', Courier, monospace;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Monitoreo End-to-End: Observabilidad del Pipeline</h1>
        <div class="summary-stats" id="stats-panel"></div>
    </div>
    
    <div class="flame-container">
        <h2 style="margin-top: 0; color: #9ca3af; font-size: 16px;">Cronograma de Spans (Flame Graph)</h2>
        <div id="flamegraph-body"></div>
    </div>

    <div class="details-panel" id="details-panel">
        <h3 id="details-title">Detalles del Span</h3>
        <div>
            <strong>Duración:</strong> <span id="details-duration"></span> ms<br>
            <strong>Tokens:</strong> <span id="details-tokens"></span><br>
            <strong>Coste:</strong> <span id="details-cost"></span><br>
            <strong>Errores:</strong> <span id="details-error" style="color: #ef4444;">Ninguno</span>
        </div>
        <h4>Argumentos de Entrada (Inputs)</h4>
        <pre id="details-inputs"></pre>
        <h4>Resultados de Salida (Outputs)</h4>
        <pre id="details-outputs"></pre>
        <h4>Metadatos Adicionales</h4>
        <pre id="details-metadata"></pre>
    </div>

    <script>
        const traceData = {trace_json};
        
        function render() {{
            if (traceData.length === 0) return;
            
            // 1. Encontrar tiempos limites para normalizar posiciones
            let minTime = Infinity;
            let maxTime = -Infinity;
            let totalTokens = 0;
            let totalCost = 0;
            let totalSpans = 0;
            
            function scanTimes(span) {{
                totalSpans++;
                if (span.start_time < minTime) minTime = span.start_time;
                if (span.end_time && span.end_time > maxTime) maxTime = span.end_time;
                
                if (span.metadata) {{
                    totalTokens += (span.metadata.input_tokens || 0) + (span.metadata.output_tokens || 0);
                    totalCost += (span.metadata.cost || 0);
                }}
                
                span.children.forEach(scanTimes);
            }}
            traceData.forEach(scanTimes);
            
            const totalDuration = maxTime - minTime;
            
            // Imprimir estadísticas
            document.getElementById('stats-panel').innerHTML = `
                <div class="stat-card">Duración Total: <span class="stat-val">${{(totalDuration * 1000).toFixed(2)}} ms</span></div>
                <div class="stat-card">Total Spans: <span class="stat-val">${{totalSpans}}</span></div>
                <div class="stat-card">Tokens Procesados: <span class="stat-val">${{totalTokens}}</span></div>
                <div class="stat-card">Coste Total: <span class="stat-val">$${{totalCost.toFixed(6)}}</span></div>
            `;
            
            // 2. Renderizar filas recursivamente con indentacion vertical
            const body = document.getElementById('flamegraph-body');
            
            function buildBarHTML(span) {{
                const startOffset = span.start_time - minTime;
                const duration = (span.end_time || span.start_time) - span.start_time;
                
                const leftPercent = totalDuration > 0 ? (startOffset / totalDuration * 100) : 0;
                const widthPercent = totalDuration > 0 ? (duration / totalDuration * 100) : 100;
                
                const isError = span.error ? 'error-bar' : '';
                const durationMs = (duration * 1000).toFixed(2);
                
                const spanStr = JSON.stringify(span).replace(/'/g, "&apos;");
                
                let html = `
                    <div class="span-row">
                        <div class="span-bar ${{isError}}" 
                             style="left: ${{leftPercent}}%; width: ${{widthPercent}}%;"
                             onclick='showDetails(${{spanStr}})'>
                             ${{span.name}} (${{durationMs}}ms)
                        </div>
                    </div>
                `;
                
                span.children.forEach(child => {{
                    html += buildBarHTML(child);
                }});
                
                return html;
            }}
            
            let graphHTML = '';
            traceData.forEach(root => {{
                graphHTML += buildBarHTML(root);
            }});
            body.innerHTML = graphHTML;
        }}
        
        function showDetails(span) {{
            const panel = document.getElementById('details-panel');
            panel.style.display = 'block';
            
            document.getElementById('details-title').innerText = "Detalles del Span: " + span.name;
            const duration = ((span.end_time || span.start_time) - span.start_time) * 1000;
            document.getElementById('details-duration').innerText = duration.toFixed(2);
            
            const inTok = span.metadata?.input_tokens || 0;
            const outTok = span.metadata?.output_tokens || 0;
            document.getElementById('details-tokens').innerText = `Entrada: ${{inTok}} | Salida: ${{outTok}} (Total: ${{inTok + outTok}})`;
            
            const cost = span.metadata?.cost || 0.0;
            document.getElementById('details-cost').innerText = "$" + cost.toFixed(6);
            
            const errSpan = document.getElementById('details-error');
            if (span.error) {{
                errSpan.innerText = span.error;
                errSpan.style.color = '#ef4444';
            }} else {{
                errSpan.innerText = 'Ninguno';
                errSpan.style.color = '#10b981';
            }}
            
            document.getElementById('details-inputs').innerText = JSON.stringify(span.inputs, null, 2);
            document.getElementById('details-outputs').innerText = JSON.stringify(span.outputs, null, 2);
            document.getElementById('details-metadata').innerText = JSON.stringify(span.metadata, null, 2);
            
            // Scroll suave a detalles
            panel.scrollIntoView({{ behavior: 'smooth' }});
        }}
        
        window.onload = render;
    </script>
</body>
</html>
"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_template)
        logger.info(f"Visualizador interactivo Flame Graph exportado exitosamente a '{filepath}'")


# Instancia única global (Singleton) para registro directo
tracer = GlobalTracer()


def trace_span(name: Optional[str] = None) -> Callable[..., Any]:
    """
    Decorador para envolver funciones y registrar automáticamente sus entradas,
    salidas, latencias y posibles excepciones dentro de la jerarquía de Spans.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        span_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Reconstruir mapa de entradas
            inputs = {}
            # Capturamos kwargs y args básicos
            if kwargs:
                inputs.update(kwargs)
            if args:
                # Los mapeamos por índice si no hay nombres disponibles
                for idx, arg in enumerate(args):
                    inputs[f"arg_{idx}"] = str(arg)
                    
            # Iniciar span
            tracer.start_span(name=span_name, inputs=inputs)
            try:
                result = func(*args, **kwargs)
                outputs = {"result": result} if result is not None else {}
                tracer.end_span(outputs=outputs)
                return result
            except Exception as e:
                tracer.end_span(error=e)
                raise e
                
        return wrapper
    return decorator
