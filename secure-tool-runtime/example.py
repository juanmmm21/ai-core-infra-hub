from runtime import SecureToolRuntime


def run_demo() -> None:
    print("=" * 70)
    print("      Demostracion de Secure Tool Runtime (Code Sandbox)      ")
    print("=" * 70)
    
    runtime = SecureToolRuntime()
    
    # 1. Codigo Seguro: Filtrar numeros y calcular media
    safe_code = """
numeros = [12, 45, 78, 23, 56, 89, 4, 15]
filtrados = [n for n in numeros if n > 40]
media = sum(filtrados) / len(filtrados)
print(f"Numeros mayores que 40: {filtrados}")
print(f"Media aritmetica: {media:.2f}")
"""

    print("\n--- CASO 1: Ejecucion de Codigo Seguro ---")
    print(f"Codigo:\n{safe_code}")
    success, stdout, stderr, err = runtime.run_code(safe_code)
    print(f"  -> EXITO: {success}")
    print(f"  -> STDOUT:\n{stdout}")
    print(f"  -> ERR: {err}\n")
    
    # 2. Codigo Inseguro: Intentar leer archivos locales usando builtins bloqueados
    unsafe_code = """
print("Intentando leer /etc/passwd...")
with open("/etc/passwd", "r") as f:
    print(f.read())
"""

    print("--- CASO 2: Bloqueo de Codigo Inseguro (AST Validation) ---")
    print(f"Codigo:\n{unsafe_code}")
    success, stdout, stderr, err = runtime.run_code(unsafe_code)
    print(f"  -> EXITO: {success}")
    print(f"  -> STDOUT: '{stdout}'")
    print(f"  -> ERR: {err}\n")
    
    # 3. Codigo con bucle infinito: Prevencion de denegacion de servicio (Timeout)
    timeout_code = """
print("Iniciando bucle infinito de calculo...")
x = 0
while True:
    x += 1
"""

    print("--- CASO 3: Control de Recursos y Bucles Infinitos (Timeout) ---")
    print(f"Codigo:\n{timeout_code}")
    # Limite estricto de 1.0 segundos
    success, stdout, stderr, err = runtime.run_code(timeout_code, timeout_seconds=1.0)
    print(f"  -> EXITO: {success}")
    print(f"  -> STDOUT: '{stdout}'")
    print(f"  -> ERR: {err}\n")


if __name__ == "__main__":
    run_demo()
