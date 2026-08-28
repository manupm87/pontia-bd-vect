# Sesión práctica 1 - Configuración del entorno

Antes de ponernos a trabajar, tendremos que equiparnos convenientemente. El proyecto utiliza **Python 3.12**, pero no es necesario instalarlo ni mantenerlo manualmente. Para preparar el entorno utilizaremos `uv`, una herramienta que se encargará de descargar la versión correcta de Python, crear un entorno virtual aislado e instalar las dependencias exactas definidas para la sesión.

## Índice de contenidos

1. [Configuración en macOS](#1-configuración-en-macos)
2. [Configuración en Linux](#2-configuración-en-linux)
3. [Configuración en Windows](#3-configuración-en-windows)

## 1. Configuración en macOS

Abre un nuevo terminal y sitúate dentro de la carpeta `sesion_01`. Sustituye la ruta del ejemplo por aquella en la que hayas guardado el material:

```bash
cd "/ruta/al/material/sesion_01"
```

Puedes comprobar que te encuentras en el directorio correcto ejecutando:

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

El instalador indicará en qué directorio ha dejado el ejecutable. Para que la terminal actual pueda encontrarlo sin necesidad de cerrarla, actualiza temporalmente el `PATH`:

```bash
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
```

Comprueba que la instalación ha terminado correctamente:

```bash
uv --version
```

Si el comando continúa sin encontrarse, cierra el terminal, vuelve a abrirla y regresa a la carpeta `sesion_01` antes de continuar.

### 1.2. Creación del entorno

El proyecto incluye un script que instala Python 3.12, crea el entorno virtual, instala todas las dependencias, registra el kernel de Jupyter y ejecuta una validación inicial:

```bash
bash scripts/setup.sh
```

La primera ejecución puede tardar varios minutos porque `uv` debe descargar Python y todas las librerías necesarias. Al terminar, la carpeta del proyecto contendrá un directorio `.venv`. No debes activar este entorno manualmente: los comandos `uv run` se encargan de utilizarlo.

Ejecuta la validación una segunda vez para confirmar que el entorno responde correctamente:

```bash
uv run python scripts/validate_content.py --quick
```

El proceso debe terminar mostrando un mensaje que comience por `Validation OK`.

### 1.3. Configuración de las variables de entorno

Durante la instalación se crea un archivo `.env` a partir de `.env.example`. Ábrelo con cualquier editor de texto. Si utilizas Visual Studio Code, puedes hacerlo desde la terminal mediante:

```bash
code .env
```

También puedes abrirlo con la aplicación de edición que prefieras. Su contenido tendrá esta estructura:

```dotenv
OPENAI_API_KEY=
COHERE_API_KEY=
GEMINI_API_KEY=
HF_TOKEN=

OPENAI_EMBEDDING_MODEL=text-embedding-3-small
COHERE_EMBEDDING_MODEL=embed-v4.0
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
LOCAL_EMBEDDING_MODEL=intfloat/multilingual-e5-small
```

Introduce cada credencial después del signo `=` sin añadir comillas ni espacios. Por ejemplo:

```dotenv
OPENAI_API_KEY=tu_clave_de_openai
```

Las variables `OPENAI_API_KEY`, `COHERE_API_KEY` y `GEMINI_API_KEY` habilitan las comparativas con sus respectivos proveedores. `HF_TOKEN` solo es necesario para acceder a modelos de Hugging Face que requieran autenticación; puede permanecer vacío para el modelo local empleado en la sesión. No modifiques los nombres de las variables ni subas el archivo `.env` a un repositorio.

### 1.4. Inicio de JupyterLab

Inicia JupyterLab desde la carpeta `sesion_01`:

```bash
uv run jupyter lab notebooks
```

El navegador mostrará el contenido del directorio `notebooks`. Abre `sesion_01_buscador_semantico_ecommerce.ipynb` y comprueba que el kernel seleccionado sea **Python (BBDD Vectoriales · Sesión 1)**. Si aparece otro kernel, selecciónalo desde el menú **Kernel → Change Kernel**.

Para detener JupyterLab, vuelve a la terminal y pulsa `Control + C`. Confirma el cierre si la terminal lo solicita.

## 2. Configuración en Linux

Abre una terminal y sitúate dentro de la carpeta `sesion_01`. Sustituye la ruta del ejemplo por aquella en la que hayas guardado el material:

```bash
cd "/ruta/al/material/sesion_01"
```

Comprueba que te encuentras en el directorio correcto:

```bash
pwd
ls
```

Entre los archivos mostrados deberían aparecer `pyproject.toml`, `uv.lock`, `notebooks` y `scripts`.

### 2.1. Instalación de `curl` y `uv`

El instalador de `uv` utiliza `curl`. En Ubuntu, Debian y distribuciones derivadas puedes instalarlo mediante:

```bash
sudo apt update
sudo apt install -y curl
```

En Fedora, utiliza:

```bash
sudo dnf install -y curl
```

En Arch Linux y distribuciones derivadas, utiliza:

```bash
sudo pacman -S curl
```

Instala después `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Actualiza el `PATH` de la terminal actual y comprueba la instalación:

```bash
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
uv --version
```

Si el comando continúa sin encontrarse, cierra la terminal, vuelve a abrirla y regresa a la carpeta `sesion_01` antes de continuar.

### 2.2. Creación del entorno

Ejecuta el script de configuración incluido en el proyecto:

```bash
bash scripts/setup.sh
```

El script descarga Python 3.12, crea `.venv`, instala las dependencias bloqueadas en `uv.lock`, registra el kernel de Jupyter y valida el material. No necesitas instalar Python mediante el gestor de paquetes de tu distribución ni activar `.venv` manualmente.

Comprueba la instalación ejecutando:

```bash
uv run python scripts/validate_content.py --quick
```

El proceso debe terminar mostrando un mensaje que comience por `Validation OK`.

### 2.3. Configuración de las variables de entorno

El script crea un archivo `.env` a partir de `.env.example`. Puedes editarlo desde la terminal con `nano`:

```bash
nano .env
```

Su contenido tendrá esta estructura:

```dotenv
OPENAI_API_KEY=
COHERE_API_KEY=
GEMINI_API_KEY=
HF_TOKEN=

OPENAI_EMBEDDING_MODEL=text-embedding-3-small
COHERE_EMBEDDING_MODEL=embed-v4.0
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
LOCAL_EMBEDDING_MODEL=intfloat/multilingual-e5-small
```

Introduce cada credencial después del signo `=` sin añadir comillas ni espacios:

```dotenv
COHERE_API_KEY=tu_clave_de_cohere
```

Las variables `OPENAI_API_KEY`, `COHERE_API_KEY` y `GEMINI_API_KEY` habilitan las comparativas con sus respectivos proveedores. `HF_TOKEN` solo es necesario para acceder a modelos de Hugging Face que requieran autenticación; puede permanecer vacío para el modelo local empleado en la sesión. Conserva sin cambios los nombres de las variables y no subas `.env` a ningún repositorio.

En `nano`, guarda el archivo pulsando `Control + O`, confirma el nombre con `Enter` y sal mediante `Control + X`.

### 2.4. Inicio de JupyterLab

Inicia JupyterLab desde la carpeta `sesion_01`:

```bash
uv run jupyter lab notebooks
```

El navegador mostrará el contenido del directorio `notebooks`. Abre `sesion_01_buscador_semantico_ecommerce.ipynb` y comprueba que el kernel seleccionado sea **Python (BBDD Vectoriales · Sesión 1)**. Si aparece otro kernel, selecciónalo desde el menú **Kernel → Change Kernel**.

Para detener JupyterLab, vuelve a la terminal y pulsa `Control + C`. Confirma el cierre si la terminal lo solicita.

## 3. Configuración en Windows

Abre **PowerShell** y sitúate dentro de la carpeta `sesion_01`. Sustituye la ruta del ejemplo por aquella en la que hayas guardado el material:

```powershell
Set-Location "C:\ruta\al\material\sesion_01"
```

Comprueba que te encuentras en el directorio correcto:

```powershell
Get-Location
Get-ChildItem
```

Entre los archivos mostrados deberían aparecer `pyproject.toml`, `uv.lock`, `notebooks` y `scripts`.

### 3.1. Instalación de `uv`

Instala `uv` mediante su instalador oficial para PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
```

Añade las ubicaciones habituales del ejecutable al `PATH` de la sesión actual y comprueba la instalación:

```powershell
$env:Path = "$HOME\.local\bin;$HOME\.cargo\bin;$env:Path"
uv --version
```

Si PowerShell continúa sin reconocer el comando, cierra la ventana, abre una nueva y regresa a la carpeta `sesion_01` antes de continuar.

### 3.2. Creación del entorno

Ejecuta el script de configuración de Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
```

El script descarga Python 3.12, crea `.venv`, instala las dependencias bloqueadas en `uv.lock`, registra el kernel de Jupyter y valida el material. No necesitas instalar Python desde Microsoft Store ni activar el entorno virtual manualmente.

Comprueba la instalación mediante:

```powershell
uv run python scripts/validate_content.py --quick
```

El proceso debe terminar mostrando un mensaje que comience por `Validation OK`.

### 3.3. Configuración de las variables de entorno

Durante la instalación se crea un archivo `.env` a partir de `.env.example`. Ábrelo con el Bloc de notas:

```powershell
notepad .env
```

El archivo tendrá esta estructura:

```dotenv
OPENAI_API_KEY=
COHERE_API_KEY=
GEMINI_API_KEY=
HF_TOKEN=

OPENAI_EMBEDDING_MODEL=text-embedding-3-small
COHERE_EMBEDDING_MODEL=embed-v4.0
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
LOCAL_EMBEDDING_MODEL=intfloat/multilingual-e5-small
```

Introduce cada credencial después del signo `=` sin añadir comillas ni espacios:

```dotenv
GEMINI_API_KEY=tu_clave_de_gemini
```

Las variables `OPENAI_API_KEY`, `COHERE_API_KEY` y `GEMINI_API_KEY` habilitan las comparativas con sus respectivos proveedores. `HF_TOKEN` solo es necesario para acceder a modelos de Hugging Face que requieran autenticación; puede permanecer vacío para el modelo local empleado en la sesión. No modifiques los nombres de las variables ni compartas o subas a un repositorio el archivo `.env`.

Guarda los cambios y cierra el Bloc de notas antes de iniciar JupyterLab.

### 3.4. Inicio de JupyterLab

Inicia JupyterLab desde la carpeta `sesion_01`:

```powershell
uv run jupyter lab notebooks
```

El navegador mostrará el contenido del directorio `notebooks`. Abre `sesion_01_buscador_semantico_ecommerce.ipynb` y comprueba que el kernel seleccionado sea **Python (BBDD Vectoriales - Sesion 1)**. Si aparece otro kernel, selecciónalo desde el menú **Kernel → Change Kernel**.

Para detener JupyterLab, vuelve a PowerShell y pulsa `Control + C`. Confirma el cierre si PowerShell lo solicita.
