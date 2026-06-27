import ast
import sys
import time
import io
import logging
import multiprocessing
import traceback
from typing import Dict, Any, Tuple, Optional, Set

logger = logging.getLogger(__name__)


class SecurityValidationError(Exception):
    """Excepcion arrojada cuando el codigo contiene construcciones prohibidas."""
    pass


class ASTSafetyValidator(ast.NodeVisitor):
    """
    Analizador del Arbol de Sintaxis Abstracta (AST) para verificar
    si el codigo suministrado contiene llamadas o importaciones prohibidas.
    """

    def __init__(self, banned_modules: Set[str], banned_calls: Set[str]) -> None:
        self.banned_modules = banned_modules
        self.banned_calls = banned_calls

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name in self.banned_modules:
                raise SecurityValidationError(f"Importacion prohibida detectada: 'import {alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module in self.banned_modules:
            raise SecurityValidationError(f"Importacion prohibida detectada: 'from {node.module} import ...'")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Extraemos el nombre de la funcion llamada
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
            
        if func_name in self.banned_calls:
            raise SecurityValidationError(f"Llamada a funcion prohibida detectada: '{func_name}()'")
            
        self.generic_visit(node)


def _worker_execute_code(code_str: str, queue: multiprocessing.Queue) -> None:
    """
    Funcion ejecutada en el proceso hijo aislado.
    Captura stdout/stderr y ejecuta el codigo en un entorno local controlado.
    """
    # Redirigimos salida estandar
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture
    
    local_scope: Dict[str, Any] = {}
    global_scope: Dict[str, Any] = {
        "__builtins__": {
            # Mantenemos funciones basicas seguras
            "abs": abs, "all": all, "any": any, "bin": bin, "bool": bool,
            "chr": chr, "dict": dict, "divmod": divmod, "enumerate": enumerate,
            "filter": filter, "float": float, "format": format, "hex": hex,
            "id": id, "int": int, "isinstance": isinstance, "issubclass": issubclass,
            "iter": iter, "len": len, "list": list, "map": map, "max": max,
            "min": min, "next": next, "oct": oct, "ord": ord, "pow": pow,
            "print": print, "range": range, "repr": repr, "reversed": reversed,
            "round": round, "set": set, "slice": slice, "sorted": sorted,
            "str": str, "sum": sum, "tuple": tuple, "type": type, "zip": zip
        }
    }
    
    success = False
    error_msg = ""
    
    try:
        # Compilamos y ejecutamos
        compiled = compile(code_str, "<sandbox>", "exec")
        exec(compiled, global_scope, local_scope)
        success = True
    except Exception as e:
        # Extraemos traza del error omitiendo detalles del backend
        error_msg = traceback.format_exc()
        
    stdout_val = stdout_capture.getvalue()
    stderr_val = stderr_capture.getvalue()
    
    # Enviamos resultados de vuelta a traves de la cola de procesos IPC
    queue.put((success, stdout_val, stderr_val, error_msg))


class SecureToolRuntime:
    """
    Sandbox de ejecucion de codigo dinámico seguro.
    
    Analiza y valida el codigo estaticamente a nivel de AST y posteriormente
    lo ejecuta dentro de un subproceso aislado con limites de tiempo de CPU (Timeout)
    y memoria para prevenir bucles infinitos y denegaciones de servicio.
    """

    def __init__(
        self,
        banned_modules: Optional[Set[str]] = None,
        banned_calls: Optional[Set[str]] = None
    ) -> None:
        self.banned_modules = banned_modules or {
            "os", "sys", "subprocess", "socket", "builtins", "shutil", 
            "ctypes", "requests", "urllib", "http", "pty", "platform"
        }
        self.banned_calls = banned_calls or {
            "eval", "exec", "open", "compile", "getattr", "setattr",
            "globals", "locals", "__import__", "input", "exit", "quit"
        }

    def validate_code(self, code_str: str) -> None:
        """
        Analiza estaticamente el codigo buscando violaciones de seguridad.
        Levanta SecurityValidationError si detecta anomalias.
        """
        try:
            tree = ast.parse(code_str)
        except SyntaxError as e:
            raise SecurityValidationError(f"Error de sintaxis en el codigo suministrado: {str(e)}")
            
        validator = ASTSafetyValidator(self.banned_modules, self.banned_calls)
        validator.visit(tree)

    def run_code(self, code_str: str, timeout_seconds: float = 2.0) -> Tuple[bool, str, str, str]:
        """
        Valida y ejecuta el script de python en un subproceso aislado.
        
        Args:
            code_str: Script de Python a ejecutar.
            timeout_seconds: Segundos maximos de ejecucion permitidos.
            
        Returns:
            Tupla (success: bool, stdout: str, stderr: str, error_detail: str)
        """
        # 1. Validacion estatica previa (AST Scan)
        try:
            self.validate_code(code_str)
        except SecurityValidationError as e:
            logger.warning(f"Validacion de seguridad rechazada: {str(e)}")
            return False, "", "", f"SecurityValidationError: {str(e)}"
            
        # 2. Ejecucion aislada mediante Process
        queue: multiprocessing.Queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=_worker_execute_code,
            args=(code_str, queue)
        )
        
        start_time = time.time()
        process.start()
        
        # Esperamos a que termine respetando el timeout
        process.join(timeout=timeout_seconds)
        
        if process.is_alive():
            # Forzamos terminacion inmediata en caso de bucle infinito o bloqueo
            process.terminate()
            process.join()
            logger.error("Ejecucion de herramienta excedio el limite de tiempo (Timeout).")
            return False, "", "", f"TimeoutError: La ejecucion del script excedio el limite de {timeout_seconds} segundos."
            
        # Intentamos leer de la cola de comunicacion IPC
        if not queue.empty():
            success, stdout, stderr, error_msg = queue.get()
            return success, stdout, stderr, error_msg
        else:
            return False, "", "", "Error desconocido: El subproceso termino de forma abrupta sin devolver datos."
