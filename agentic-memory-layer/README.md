# Agentic Memory Layer

Capa de memoria persistente y estructurada para agentes de Inteligencia Artificial que remedia la limitacion de contexto de los modelos de lenguaje (amnesia) mediante una arquitectura hibrida: un buffer de corto plazo conversacional con sintesis automatica y un almacen episódico de largo plazo con decaimiento temporal basado en la curva del olvido de Ebbinghaus.

## Arquitectura de Memoria Hibrida

El modulo implementa un sistema coordinado de almacenamiento en dos niveles para equilibrar la velocidad de recuperacion en el dialogo activo con la retencion historica de hechos relevantes.

```mermaid
graph TD
    Input[Entrada del Agente / Turno de Conversacion] --> ShortTerm[Memoria de Corto Plazo: Conversational Buffer]
    
    subgraph Memoria Corto Plazo
        ShortTerm --> LengthCheck{¿Turnos > Limite Maximo?}
        LengthCheck -->|Si| Summarize[Extraer Hechos & Compactar]
        LengthCheck -->|No| KeepActive[Conservar Turnos Intactos]
        Summarize --> SaveLong[Persistir Hecho en Largo Plazo]
        Summarize --> SummaryBuffer[Historial de Resumen Acumulado]
    end
    
    subgraph Memoria Largo Plazo (Episodica)
        SaveLong --> VectorGen[Generador de Embeddings local]
        VectorGen --> StorageCheck{¿NanoVectorDB disponible?}
        StorageCheck -->|Si| NanoVectorDB[HNSW Index en agent_long_term_memory.json]
        StorageCheck -->|No| MemoryList[Fallback: Lista en Memoria]
    end
    
    Query[Consulta / Busqueda del Agente] --> RecallFlow[Bucle de Recuperacion - Recall]
    RecallFlow --> SearchVectors[Busqueda de Vectores Cercanos]
    SearchVectors --> Ebbinghaus[Aplicar Curva de Olvido Exponencial]
    Ebbinghaus --> RankResults[Top-K Recuerdos mas Importantes]
```

### 1. Memoria de Corto Plazo (Conversational Buffer)

Conserva las ultimas interacciones en su formato original de mensaje (`role` y `content`).
*   **Limite de Capacidad:** El buffer acepta un numero maximo de turnos ($N_{\text{max}} = 5$ interacciones dobles, es decir, 10 mensajes en total).
*   **Algoritmo de Sintesis Automatica:** Al rebasar el limite, el sistema extrae de forma heurística los hechos clave de los turnos antiguos, dejando los ultimos 2 turnos (los mas recientes) intactos en el buffer. La informacion resumida se anexa a la cadena `conversation_summary`, la cual se antepone en el prompt del agente, liberando espacio util en la ventana de contexto sin romper el hilo conductor de la sesion.

### 2. Memoria de Largo Plazo (Episodic Decay Memory)

Almacena hechos de forma permanente asociandoles un vector de caracteristicas (embedding), un timestamp de creacion y un factor de relevancia subjetiva.

#### La Curva del Olvido Exponencial (Heuristica de Ebbinghaus)
Al realizar una busqueda de recuerdos para responder a una consulta del usuario, el sistema no se limita a evaluar la similitud semantica vectorial. Aplica una formula de decaimiento temporal exponencial de la relevancia para imitar la memoria biologica:

$$\text{Recall Score}(F) = \text{Sim}_{\cos}(Q, F) \cdot \text{Imp}(F) \cdot e^{-\lambda \cdot \Delta t}$$

Donde:
*   $\text{Sim}_{\cos}(Q, F)$ es la similitud de coseno proyectada al rango $[0, 1]$ entre el vector de la consulta $Q$ y el vector del recuerdo indexado $F$:
    $$\text{Sim}_{\cos}(Q, F) = \frac{\cos(\mathbf{q}, \mathbf{f}) + 1.0}{2.0}$$
*   $\text{Imp}(F) \in [1, 10]$ es el peso de importancia asignado originalmente al hecho al ser almacenado.
*   $\lambda$ es el coeficiente de velocidad de olvido (`decay_factor`, por defecto `0.05`).
*   $\Delta t = t_{\text{consulta}} - t_{\text{creacion\_hecho}}$ representa la antiguedad del recuerdo en segundos.

Bajo este modelo, recuerdos antiguos con baja importancia son "olvidados" (su score tiende a cero), mientras que hechos criticos con alta importancia o recuerdos extremadamente recientes y semanticamente afines dominan el ranking.

#### Generador Determinista de Embeddings (Word-Hash Local)
Para operar offline sin dependencias externas, el modulo calcula localmente un vector unitario de dimension $D$:
1.  Descompone el texto del hecho en palabras utilizando la expresion regular `\w+`.
2.  Genera el hash MD5 de cada palabra y calcula el indice modular correspondiente:
    $$h = \text{int(MD5}(w)\text{, 16)} \pmod D$$
3.  Incrementa el valor del vector en la posicion $h$: $V_h \leftarrow V_h + 1.0$.
4.  Aplica normalizacion L2 al vector resultante para proyectarlo a la superficie de una hiperesfera unitaria, permitiendo que el producto escalar represente la similitud de coseno exacta.

## Conexión con el Ecosistema

Este componente se conecta con los modulos de la infraestructura de la siguiente forma:
1.  **nano-vector-db:** Actua como el almacen fisico de la memoria episódica. Al invocar `save_fact`, la memoria guarda el vector y sus metadatos usando `NanoVectorDB.insert(id, vector, metadata)` y persiste los datos en disco bajo el archivo `agent_long_term_memory.json`.
2.  **orchestra-agents:** Cada agente autonomo (Researcher, Writer) utiliza una instancia de esta memoria para persistir su scratchpad de ejecucion ReAct y recuperar informacion contextual historica de ejecuciones pasadas.

## Estructura del Proyecto

*   `memory.py`: Implementacion de la clase `AgenticMemory`, el buffer de corto plazo con algoritmo de sintesis, el hasheador local de embeddings y el motor de puntuacion temporal de Ebbinghaus.
*   `test_memory.py`: Suite de test unitarios asincronos que validan la tasa de compactacion del buffer, la consistencia de los embeddings hash y la precision de la tasa de decaimiento exponencial frente al tiempo transcurrido.
*   `example.py`: Demostracion interactiva de simulacion de paso del tiempo (30 segundos y 10 minutos) que ilustra de que manera se atenua el score de los recuerdos viejos y cuales son retenidos por el agente.

## Instalacion y Uso

### 1. Inicializar el Entorno e Instalar Dependencias

Navegue al directorio e instale las dependencias locales requeridas:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Ejecutar Pruebas Unitarias

La suite valida que la logica de decaimiento temporal y la persistencia HNSW coincidan exactamente con la teoria:

```bash
.venv/bin/python -m unittest test_memory.py
```

### 3. Ejecutar Demostración de Curva de Olvido

Ejecute la demostracion para visualizar como decaen las puntuaciones de recuerdos a traves del tiempo:

```bash
.venv/bin/python example.py
```
