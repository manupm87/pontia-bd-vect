# Sesión práctica 3 · De un índice vectorial a una base de datos vectorial

En la sesión 1 construimos e interpretamos el espacio vectorial. En la sesión 2 aprendimos a localizar vecinos exactos y aproximados. Esta tercera sesión añade la capa que faltaba: un sistema que conserva vectores **y** documentos, acepta filtros y mutaciones, persiste el estado, expone una API y asume responsabilidades operativas.

El material es autocontenido. Incluye el mismo snapshot de 50.000 productos españoles, los mismos embeddings E5 normalizados y las 276 consultas de la sesión anterior. Así podemos comparar cinco motores sin cambiar simultáneamente el modelo o el conjunto de evaluación.

## Guía de contenido

Los notebooks están numerados y se ejecutan en este orden:

1. `sesion_03_00_del_indice_a_la_bbdd_vectorial.ipynb`: marco conceptual y contrato común.
2. `sesion_03_01_pinecone_cloud.ipynb`: camino gestionado principal.
3. `sesion_03_02_chroma.ipynb`: laboratorio local con Chroma.
4. `sesion_03_03_weaviate.ipynb`: laboratorio local con Weaviate.
5. `sesion_03_04_milvus.ipynb`: laboratorio local con Milvus Standalone.
6. `sesion_03_05_qdrant.ipynb`: laboratorio local con Qdrant.
7. `sesion_03_06_langchain_vectorstores.ipynb`: Chroma y Qdrant como vector stores de LangChain, sin LLM.

El recorrido mínimo de clase es el notebook conceptual, Pinecone Cloud, **uno** de los cuatro laboratorios locales y LangChain. Los demás quedan como alternativas equivalentes. Los cinco laboratorios principales emplean SDKs nativos; LangChain llega después, cuando ya sabemos qué contrato intenta abstraer y qué detalles no logra homogeneizar.

> _La memoria ampliada se encuentra en [MEMORIA_SESION_03.md](MEMORIA_SESION_03.md) y [MEMORIA_SESION_03.pdf](MEMORIA_SESION_03.pdf). Las fuentes primarias y versiones utilizadas están recopiladas en [docs/REFERENCIAS.md](docs/REFERENCIAS.md)._

## 1. Preparar Python en macOS o Linux

Abre una terminal, entra en `sesion_03` e instala `uv` si aún no lo tienes:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
bash scripts/setup.sh
```

El script instala Python 3.12, crea `.venv`, resuelve todas las familias de SDK compatibles, registra el kernel de Jupyter, construye los notebooks y valida el material. No necesitas activar el entorno: `uv run` lo selecciona automáticamente.

## 2. Preparar Python en Windows

Abre PowerShell dentro de `sesion_03`:

```powershell
powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
$env:Path = "$HOME\.local\bin;$HOME\.cargo\bin;$env:Path"
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
```

Docker Desktop debe utilizar contenedores Linux. Si Windows solicita habilitar WSL 2, completa primero esa instalación y reinicia el equipo.

## 3. Configurar credenciales y servicios

`scripts/setup.*` copia `.env.example` como `.env`. Las claves permanecen fuera de los notebooks y `.env` está ignorado por Git.

Para Pinecone completa `PINECONE_API_KEY`. El notebook crea un índice serverless BYOV de 384 dimensiones, métrica coseno, cloud `aws`, región `us-east-1` y namespace `esci-es-s03`. La limpieza remota está desactivada: el laboratorio elimina solamente su documento canario.

Para un motor local abre otra terminal y arranca **un** compose:

```bash
docker compose -f deploy/chroma/compose.yaml up -d
docker compose -f deploy/chroma/compose.yaml ps
```

Sustituye `chroma` por `weaviate`, `milvus` o `qdrant`. Antes de cambiar de motor:

```bash
docker compose -f deploy/chroma/compose.yaml down
```

`down` detiene los contenedores pero conserva los volúmenes. Añadir `--volumes` borra la base local y, por tanto, es una operación destructiva. Los detalles y comprobaciones de salud están en [deploy/README.md](deploy/README.md).

## 4. Ejecutar JupyterLab

```bash
uv run jupyter lab notebooks
```

Selecciona **Python (BBDD Vectoriales · Sesión 3)**. Cada notebook de proveedor realiza un preflight antes de tocar el servicio. La ingesta de 50.000 registros es idempotente; si vuelves a ejecutar el notebook, los UUIDv5 estables actualizan los mismos puntos.

Los recursos tienen el prefijo `bbdd-vectoriales-s03`. `S03_ALLOW_RESET=false` evita recrear colecciones por accidente y `S03_ALLOW_REMOTE_CLEANUP=false` protege Pinecone. La limpieza final de cada notebook permanece desactivada hasta que `S03_CONFIRM_CLEANUP` contiene exactamente `DELETE:<nombre-del-recurso>`. No cambies esos controles para apuntar a un recurso compartido.
