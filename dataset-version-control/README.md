# Dataset Version Control

Utilidad de control de versiones ligera para conjuntos de datos (datasets) de aprendizaje automatico (Machine Learning), disenada para garantizar la trazabilidad y reproducibilidad cientifica en flujos de trabajo de MLOps.

Este modulo funciona de manera analoga a Git, abstrayendo el versionado mediante instantaneas (snapshots) fisicas comprimidas y almacenadas en un directorio indexado por hashes criptograficos. Esto permite realizar auditorias de cambios en los datos, retornar a versiones anteriores (checkout) y calcular diferencias semanticas a nivel de registros logicos JSON.

## Arquitectura y Flujo de Versionado de Datos

El sistema encapsula la gestion del historial estructurando un almacen local oculto en la ruta `.dvc_store/`.

```mermaid
flowchart TD
    Dataset[Dataset de Trabajo: dataset.json] --> CommitCmd[Comando: VCS.commit]
    
    subgraph Creacion de Snapshot (Almacenamiento por Contenido)
        CommitCmd --> MD5Calc[Calcular MD5 de dataset.json]
        MD5Calc --> ObjectStore{¿Existe Hash en objects/?}
        ObjectStore -->|Si - Evitar Duplicidad| SkipCopy[Deduplicar Contenido]
        ObjectStore -->|No| CopyFile[Copiar a objects/hash.dat]
    end
    
    subgraph Creacion de Commit en el Grafo
        SkipCopy & CopyFile --> Metadata[Generar CommitMetadata]
        Metadata --> SHA1[SHA-1 de Metadatos + Parent Hash]
        SHA1 --> UpdateCommits[Registrar en commits.json]
        UpdateCommits --> UpdateHEAD[Actualizar head.json]
    end
    
    CheckoutCmd[Comando: VCS.checkout] --> ReadHEAD[Leer head.json & commits.json]
    ReadHEAD --> GetObject[Localizar objects/file_hash.dat]
    GetObject --> Restore[Restaurar dataset.json en area de trabajo]
    
    DiffCmd[Comando: VCS.diff] --> ParseJSON[Deserializar JSONs de Commits A y B]
    ParseJSON --> PKDetect[Autodetectar Llave Primaria: id, guid, instruction]
    PKDetect --> CompareSets[Calcular Diferencia de Conjuntos y Atributos]
    CompareSets --> ReturnStats[Retornar Estadisticas: Agregados, Eliminados, Modificados]
```

### 1. Almacenamiento Direccionado por Contenido (CAS)

Para optimizar el uso del disco y dar soporte a datasets de gran volumen, el modulo implementa una estrategia de Deduplicacion por Contenido:

*   **Identificador MD5:** Al confirmar un archivo de datos, el sistema calcula su firma MD5:
    $$H_{\text{file}} = \text{MD5}(\text{bytes\_del\_archivo})$$
*   **Almacen de Objetos (CAS):** El archivo se copia al directorio `.dvc_store/objects/[H_file].dat`. Si el contenido del dataset no ha cambiado, el hash resultante sera identico, por lo que el sistema omite la copia fisica y registra unicamente un nuevo puntero en el historial, reduciendo el sobrecoste de almacenamiento.
*   **HEAD Pointers:** Se mantiene un puntero en `.dvc_store/head.json` con la rama activa y el commit actual.

### 2. Estructura de commits y DAG (Grafo Dirigido Aciclico)

Cada confirmacion genera un nodo de tipo `CommitMetadata` que se persiste en `.dvc_store/commits.json`. El hash unico del commit se calcula mediante la funcion SHA-1:

$$H_{\text{commit}} = \text{SHA-1}(H_{\text{parent}} + H_{\text{file}} + \text{message} + \text{timestamp})$$

Al almacenar el hash del commit padre ($H_{\text{parent}}$), se establece una cadena cronologica inmutable que permite reconstruir el historial en forma de Grafo Dirigido Acíclico (DAG) y revertir de manera limpia el area de trabajo a cualquier estado pasado.

### 3. Diferenciador Semantico JSON (Data Diffs)

A diferencia de las herramientas convencionales de software (como `git diff` o `diff`) que operan a nivel de lineas de texto plano (generando falsos positivos debido a cambios en la ordenacion de llaves, formateo de lineas o indentacion), este modulo analiza semanticamente la estructura de los datos:

1.  **Deserializacion:** Convierte los archivos JSON de los commits comparados a objetos de memoria de Python.
2.  **Mapeo por Clave Primaria:** Si el archivo es una lista estructurada, autodetecta una columna identificadora unica para cada registro, priorizando los campos `id`, `guid` e `instruction` en ese orden.
3.  **Evaluacion de Diferencias:**
    *   *Agregados:* Registros cuyas claves primarias existen en el commit de destino pero no en el de origen.
    *   *Eliminados:* Registros cuyas claves primarias existen en el commit de origen pero no en el de destino.
    *   *Modificados:* Registros que comparten clave primaria pero difieren en el valor de al menos uno de sus campos o atributos internos.

## Conexión con el Ecosistema

Este modulo actua como la capa de control de datos (Data Control Plane) para:
1.  **synthetic-data-generator:** Versiona y cataloga las salidas de datos generadas (`synthetic_instruction_dataset.json` y `synthetic_dpo_dataset.json`), permitiendo comparar semanticamente el impacto de diferentes modificaciones en las directivas mutadas.
2.  **llm-qlora-finetuner:** Asegura la reproducibilidad. Antes de lanzar el entrenamiento parametrico, el pipeline ejecuta un checkout del dataset al hash de confirmacion exacto que se desea auditar, bloqueando la deriva de datos (*data drift*).

## Estructura del Proyecto

*   `dvc.py`: Implementacion de la clase principal `DatasetVCS` y el esquema Pydantic `CommitMetadata` para gestionar commits, checkouts y diffs.
*   `test_dvc.py`: Suite de test unitarios que comprueba la inicializacion, creacion de commits, deduplicacion fisica, restauracion (checkout) e integridad del grafo.
*   `example.py`: Demostracion interactiva del ciclo de vida del versionado. Modifica registros de un dataset JSON simulando inserciones y borrados, calcula las diferencias y restaura la version inicial.

## Instalacion y Uso

### 1. Activar el Entorno Local e Instalar Dependencias

Navegue al directorio de trabajo e instale los requerimientos:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Ejecutar Pruebas de Control de Versiones

```bash
.venv/bin/python -m unittest test_dvc.py
```

### 3. Ejecutar la Demostracion de VCS

```bash
.venv/bin/python example.py
```

El script inicializara el repositorio `.dvc_store`, confirmara el archivo inicial, realizara ediciones controladas (modificaciones y borrados), emitira un reporte detallado con las diferencias de registros y finalmente revertira el archivo a su estado original para ilustrar el control de versiones.
