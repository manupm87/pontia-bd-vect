# Sesión práctica 2 - Configuración del entorno

Antes de ejecutar los índices de FAISS, tendremos que preparar un entorno aislado con **Python 3.12** y las versiones de las dependencias utilizadas en el proyecto. La instalación se realizará con `uv`, que descargará Python, creará el entorno virtual e instalará las librerías definidas en `uv.lock`.

## Índice de contenidos

1. [Configuración en macOS](#1-configuración-en-macos)
2. [Configuración en Linux](#2-configuración-en-linux)
3. [Configuración en Windows](#3-configuración-en-windows)

## 1. Configuración en macOS

Abre **Terminal** y sitúate dentro de la carpeta `sesion_02`. Sustituye la ruta del ejemplo por aquella en la que hayas guardado el material:

```bash
cd "/ruta/al/material/sesion_02"
```

Comprueba que te encuentras en el directorio correcto:

```bash
pwd
ls
```

Entre los archivos mostrados deberían aparecer `pyproject.toml`, `uv.lock`, `notebooks` y `scripts`.

### 1.1. Instalación de `uv`

Instala `uv` mediante su instalador oficial:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Actualiza el `PATH` de la terminal actual y verifica la instalación:

```bash
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
uv --version
```

Si el comando continúa sin encontrarse, cierra Terminal, vuelve a abrirla y regresa a la carpeta `sesion_02`.

### 1.2. Creación del entorno

Ejecuta el script de configuración incluido en el proyecto:

```bash
bash scripts/setup.sh
```

El script instala Python 3.12, crea `.venv`, instala las dependencias, copia la plantilla de variables de entorno, registra el kernel de Jupyter y valida el material. La primera ejecución puede tardar varios minutos. No es necesario activar `.venv`: `uv run` seleccionará automáticamente ese entorno.

Comprueba el resultado ejecutando:

```bash
uv run python scripts/validate_content.py --quick
```

El proceso debe terminar con un mensaje que comience por `Validation OK`.

### 1.3. Configuración de FAISS

El script crea un archivo `.env`. Ábrelo con el editor que prefieras. Si utilizas Visual Studio Code:

```bash
code .env
```

El archivo contiene estas variables:

```dotenv
HF_TOKEN=
LOCAL_EMBEDDING_MODEL=intfloat/multilingual-e5-small
FAISS_NUM_THREADS=1
```

`FAISS_NUM_THREADS` fija el número de threads de CPU utilizados durante los benchmarks. Mantén el valor `1` para que todas las configuraciones se midan con el mismo presupuesto. Puedes aumentarlo para experimentar con el paralelismo de tu equipo, pero los tiempos obtenidos dejarán de ser comparables con una ejecución realizada bajo otra configuración.

Los datos y embeddings necesarios ya están incluidos. `HF_TOKEN` puede permanecer vacío; solo se utiliza si decides regenerar los embeddings descargando el modelo desde Hugging Face. `LOCAL_EMBEDDING_MODEL` identifica el encoder con el que se construyó el espacio vectorial y no debe modificarse al ejecutar los artefactos existentes.

### 1.4. Inicio de JupyterLab

Inicia JupyterLab desde la carpeta `sesion_02`:

```bash
uv run jupyter lab notebooks
```

Abre `sesion_02_faiss_indices_ann.ipynb` y comprueba que el kernel seleccionado sea **Python (BBDD Vectoriales · Sesión 2)**. Si aparece otro, selecciónalo desde **Kernel → Change Kernel**.

Para detener JupyterLab, vuelve a Terminal y pulsa `Control + C`.

## 2. Configuración en Linux

Abre una terminal y sitúate dentro de la carpeta `sesion_02`:

```bash
cd "/ruta/al/material/sesion_02"
```

Comprueba el contenido del directorio:

```bash
pwd
ls
```

Deberían aparecer `pyproject.toml`, `uv.lock`, `notebooks` y `scripts`.

### 2.1. Instalación de `curl` y `uv`

En Ubuntu, Debian y distribuciones derivadas, instala `curl` mediante:

```bash
sudo apt update
sudo apt install -y curl
```

En Fedora utiliza:

```bash
sudo dnf install -y curl
```

En Arch Linux y distribuciones derivadas utiliza:

```bash
sudo pacman -S curl
```

Instala `uv`, actualiza el `PATH` y verifica que el comando esté disponible:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
uv --version
```

Si la terminal no reconoce `uv`, ciérrala, abre una nueva y regresa a `sesion_02`.

### 2.2. Creación del entorno

Ejecuta el instalador del proyecto:

```bash
bash scripts/setup.sh
```

El script descarga Python 3.12, crea `.venv`, instala FAISS y el resto de dependencias, genera `.env`, registra el kernel de Jupyter y ejecuta una validación inicial. No necesitas instalar Python con el gestor de paquetes de tu distribución ni activar manualmente el entorno virtual.

Valida la instalación:

```bash
uv run python scripts/validate_content.py --quick
```

El proceso debe terminar con un mensaje que comience por `Validation OK`.

### 2.3. Configuración de FAISS

Edita el archivo `.env` mediante `nano`:

```bash
nano .env
```

Su contenido será:

```dotenv
HF_TOKEN=
LOCAL_EMBEDDING_MODEL=intfloat/multilingual-e5-small
FAISS_NUM_THREADS=1
```

Mantén `FAISS_NUM_THREADS=1` para ejecutar todos los benchmarks con un único thread. Cambiarlo modifica el presupuesto de CPU y, por tanto, los tiempos. Los embeddings ya están incluidos, de modo que `HF_TOKEN` puede permanecer vacío. `LOCAL_EMBEDDING_MODEL` documenta el encoder del índice y no debe cambiarse mientras se reutilicen las matrices proporcionadas.

En `nano`, guarda mediante `Control + O`, confirma con `Enter` y sal con `Control + X`.

### 2.4. Inicio de JupyterLab

Inicia JupyterLab:

```bash
uv run jupyter lab notebooks
```

Abre `sesion_02_faiss_indices_ann.ipynb` y selecciona el kernel **Python (BBDD Vectoriales · Sesión 2)** desde **Kernel → Change Kernel** si no aparece activado.

Para detener JupyterLab, vuelve a la terminal y pulsa `Control + C`.

## 3. Configuración en Windows

Abre **PowerShell** y sitúate dentro de la carpeta `sesion_02`:

```powershell
Set-Location "C:\ruta\al\material\sesion_02"
```

Comprueba el contenido del directorio:

```powershell
Get-Location
Get-ChildItem
```

Deberían aparecer `pyproject.toml`, `uv.lock`, `notebooks` y `scripts`.

### 3.1. Instalación de `uv`

Instala `uv` mediante el instalador oficial para PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
```

Añade las ubicaciones habituales al `PATH` de la sesión y verifica la instalación:

```powershell
$env:Path = "$HOME\.local\bin;$HOME\.cargo\bin;$env:Path"
uv --version
```

Si PowerShell no reconoce el comando, cierra la ventana, abre una nueva y regresa a `sesion_02`.

### 3.2. Creación del entorno

Ejecuta el instalador de Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
```

El script instala Python 3.12, crea `.venv`, instala FAISS y el resto de dependencias, genera `.env`, registra el kernel de Jupyter y valida el material. No necesitas instalar Python desde Microsoft Store ni activar manualmente el entorno virtual.

Comprueba la instalación:

```powershell
uv run python scripts/validate_content.py --quick
```

El proceso debe terminar con un mensaje que comience por `Validation OK`.

### 3.3. Configuración de FAISS

Abre `.env` con el Bloc de notas:

```powershell
notepad .env
```

El archivo contiene:

```dotenv
HF_TOKEN=
LOCAL_EMBEDDING_MODEL=intfloat/multilingual-e5-small
FAISS_NUM_THREADS=1
```

Mantén `FAISS_NUM_THREADS=1` para que todos los índices utilicen el mismo presupuesto de CPU. Los embeddings necesarios están incluidos, por lo que `HF_TOKEN` puede permanecer vacío. `LOCAL_EMBEDDING_MODEL` identifica el encoder utilizado para construir las matrices y no debe modificarse al trabajar con esos artefactos.

Guarda el archivo y cierra el Bloc de notas.

### 3.4. Inicio de JupyterLab

Inicia JupyterLab:

```powershell
uv run jupyter lab notebooks
```

Abre `sesion_02_faiss_indices_ann.ipynb` y comprueba que el kernel seleccionado sea **Python (BBDD Vectoriales · Sesión 2)**. Si aparece otro, selecciónalo desde **Kernel → Change Kernel**.

Para detener JupyterLab, vuelve a PowerShell y pulsa `Control + C`.
