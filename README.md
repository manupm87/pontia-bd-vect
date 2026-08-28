# Aurum Market · Motor de descubrimiento y control de catálogo

Solución de la actividad evaluable de Bases de Datos Vectoriales: un sistema de
búsqueda semántica sobre un catálogo de 15.000 productos en español que resuelve
dos recorridos de negocio — descubrimiento por intención (con filtros de marca
ejecutados por la base de datos) y control de altas potencialmente duplicadas —
sobre **Qdrant** con su SDK nativo.

El enunciado y los datos viven en `resources/actividad_evaluable/`. El informe
final es `INFORME_AURUM_MARKET.pdf` (LaTeX académico a dos columnas, fuente en
[`docs/informe/`](docs/informe/)), los
artefactos entregables están en [`resultados/`](resultados/) y la configuración
exacta de la ejecución final en [`config/run_config.yaml`](config/run_config.yaml).

## Índice de contenidos

1. [Requisitos](#1-requisitos)
2. [Instalación](#2-instalación)
3. [Recorrido principal](#3-recorrido-principal)
4. [Variables de entorno](#4-variables-de-entorno)
5. [Estructura del repositorio](#5-estructura-del-repositorio)
6. [Comandos disponibles](#6-comandos-disponibles)
7. [Tiempos aproximados](#7-tiempos-aproximados)
8. [Solución de fallos previsibles](#8-solución-de-fallos-previsibles)
9. [Seguridad operativa](#9-seguridad-operativa)

## 1. Requisitos

- Linux, macOS o Windows con WSL2.
- [uv](https://docs.astral.sh/uv/) (el script de setup lo instala si falta).
- Docker con Docker Compose (para Qdrant).
- ~3 GB de disco para el entorno (PyTorch + sentence-transformers) y ~1.5 GB
  para los modelos de embeddings descargados de Hugging Face.
- No se necesita GPU; con GPU la generación de embeddings baja de ~6 min a <1 min.

## 2. Instalación

```bash
make setup        # instala uv si falta, sincroniza el entorno y copia .env.example a .env
```

En Windows sin WSL2: `powershell -ExecutionPolicy ByPass -File scripts/setup.ps1`.

No hace falta activar el entorno manualmente: todos los comandos van vía `uv run`.

## 3. Recorrido principal

Preparar entorno, levantar el motor, ingerir, evaluar y limpiar:

```bash
make setup        # 1. entorno Python
make up           # 2. Qdrant en Docker (espera al healthcheck)
make embeddings   # 3. genera los embeddings de las 4 configuraciones locales
make pipeline     # 4. ingesta + experimentos + duplicados + métricas + resultados + eventos
make down         # 5. detiene Qdrant (conserva el volumen)
```

`make pipeline` encadena, en este orden: `embeddings` (se omite si ya existen),
`ingest` (ingesta idempotente por
lotes con verificación de recuento e indexación), `experiments` (comparativa de
representaciones frente a un oráculo exacto), `duplicates` (calibración de la
regla en desarrollo y decisiones sobre el conjunto ciego), `evaluate`
(nDCG/Recall/MRR@10, fidelidad ANN y latencias p50/p95 →
`resultados/metricas_desarrollo.json`), `search-results` (top-10 ciego y
consultas filtradas → `resultados/resultados_busqueda.csv`) y `events`
(24 eventos aplicados dos veces con verificación de idempotencia y visibilidad).

Las métricas se regeneran desde un único comando: `make metrics`.

### Los notebooks de I+D

La justificación de cada decisión vive en una serie de seis notebooks
ejecutables contra el sistema vivo (entregados ya ejecutados, con tablas y
gráficos), uno por bloque del enunciado:

| Notebook | Bloque |
|---|---|
| `actividad_00_caso_datos_y_baseline` | Problema, exploración de datos y baseline BM25 |
| `actividad_01_representacion` | Composición del texto, modelos, experimentos que decidieron |
| `actividad_02_indice_y_bbdd` | Esquema, HNSW explícito, ingesta idempotente, persistencia |
| `actividad_03_recuperacion_y_filtros` | Interfaz común, filtros por marca, casos límite |
| `actividad_04_operaciones_y_duplicados` | Eventos con visibilidad, regla de duplicados calibrada |
| `actividad_05_evaluacion_y_analisis` | §5 completo: métricas, fidelidad, latencia, atribución de errores y checklist |

Como en las sesiones del curso, los notebooks son artefactos generados:
`scripts/build_notebook.py` los construye celda a celda desde
`scripts/notebooks_src/` y `scripts/execute_notebook.py` los ejecuta en headless.

```bash
make lab               # abre JupyterLab (kernel "Python (Aurum Market · Actividad)")
make execute-notebook  # regenera la serie y la ejecuta entera sin interfaz (~35 s)
```

Requieren haber ejecutado antes el recorrido principal (Qdrant arriba, embeddings
e ingesta hechos). También hay consulta interactiva desde terminal:

```bash
make search q="taladro sin cable potente"
make search q="zapatillas para correr" brand=NIKE
```

## 4. Variables de entorno

Copia `.env.example` como `.env` (lo hace `make setup`). Ninguna variable
contiene secretos obligatorios: todo el recorrido evaluado es local.

| Variable | Uso |
|---|---|
| `HF_TOKEN` | Opcional. Solo si Hugging Face limita descargas anónimas. |
| `QDRANT_URL` | Endpoint de Qdrant (`http://localhost:6333` por defecto). |
| `QDRANT_API_KEY` | Vacío en local; solo para un Qdrant remoto protegido. |
| `QDRANT_COLLECTION` | Colección principal del catálogo. |
| `QDRANT_EVENTS_COLLECTION` | Colección dedicada a la prueba de eventos. |
| `GEMINI_API_KEY` | Opcional. Activa el experimento `gemini_v2_title` (sesión 01); sin clave se omite. |
| `AURUM_ALLOW_RESET` | `false` por defecto. Permite recrear colecciones desde cero. |
| `AURUM_CONFIRM_CLEANUP` | Vacío por defecto. Exige `DELETE:<coleccion>` para borrar. |

## 5. Estructura del repositorio

```
config/run_config.yaml        # configuración de la ejecución final (contrato reproducible)
deploy/qdrant/compose.yaml    # Qdrant 1.18.2 con volumen persistente y healthcheck
docs/                         # diagrama, referencias y fuente LaTeX del informe
notebooks/                    # serie de I+D 00-05, ejecutada (generada por scripts/build_notebook.py)
resources/actividad_evaluable # enunciado y datos (solo lectura)
resultados/                   # artefactos entregables
scripts/                      # pipeline por etapas (un script = una responsabilidad)
src/aurum_discovery/          # librería: datos, embeddings, almacén, búsqueda, evaluación
tests/                        # unitarios offline + integración marcada contra Qdrant
.artifacts/                   # informes intermedios regenerables (no versionado)
data/embeddings/              # matrices generadas por make embeddings (no versionado)
```

Separación por capas dentro de `src/aurum_discovery/`: `data.py` (carga y
validación del snapshot), `embeddings.py` (representación), `vector_store.py`
(almacenamiento e índice), `service.py` + `lexical.py` (búsqueda),
`evaluation.py` + `duplicates.py` (evaluación y regla de duplicados),
`operations.py` (seguridad y artefactos), `contracts.py` (tipos comunes).

## 6. Comandos disponibles

| Comando | Efecto |
|---|---|
| `make up` / `make down` | Arranca/detiene Qdrant. `make down-volumes` borra también el volumen. |
| `make embeddings` | Genera los embeddings de cada configuración con manifiesto SHA-256. |
| `make ingest` | Ingesta idempotente del catálogo completo y verificación de recuento. |
| `make experiments` | Comparativa BM25 + 4 configuraciones densas sobre desarrollo (y Gemini si hay clave). |
| `make duplicates` | Calibra la regla en desarrollo y genera `resultados_duplicados.csv`. |
| `make evaluate` / `make metrics` | Regenera `resultados/metricas_desarrollo.json`. |
| `make search-results` | Genera `resultados_busqueda.csv` y verifica los filtros de marca. |
| `make events` | Aplica los 24 eventos (dos veces), verifica el estado registro a registro y mide la visibilidad. |
| `make sweep-ef` | Barrido de `ef_search`: fidelidad ANN y latencia por valor. |
| `make evidence` | Copia la evidencia de la ejecución final a `resultados/evidencia/`. |
| `make pipeline` | Todo lo anterior en orden. |
| `make notebook` / `make execute-notebook` | Regenera la serie de notebooks y la ejecuta headless. |
| `make lab` | Abre los notebooks en JupyterLab. |
| `make test` | Pruebas unitarias offline. |
| `make test-integration` | Pruebas contra Qdrant (necesita `make up`). |
| `make verify` | Lint + formato + pruebas unitarias. |
| `make informe` | Compila el informe LaTeX (`docs/informe/`) a `INFORME_AURUM_MARKET.pdf`. |

## 7. Tiempos aproximados

Medidos en un equipo con 8 vCPU, 46 GB de RAM y GPU (RTX 5090); sin GPU,
multiplica la fase de embeddings por ~8.

| Fase | Tiempo |
|---|---|
| `make setup` | 2–4 min (descarga de PyTorch) |
| `make embeddings` (4 configuraciones locales) | ~1 min con GPU · 6–10 min con CPU |
| `make ingest` (15.000 registros) | ~30 s |
| `make experiments` | ~1 min (BM25 domina el coste) |
| `make duplicates` + `make evaluate` + `make search-results` | ~2 min |
| `make events` | ~30 s |

## 8. Solución de fallos previsibles

- **`No se puede conectar con Qdrant`** — el servicio no está arriba: `make up`.
  Comprueba el healthcheck con `docker compose -f deploy/qdrant/compose.yaml ps`.
- **`No existen embeddings para ...`** — falta `make embeddings` (o se borró
  `data/embeddings/`).
- **`La colección ... está vacía`** — falta `make ingest`.
- **Descarga de modelos bloqueada (HTTP 429)** — define `HF_TOKEN` en `.env`.
- **`El esquema de ... no coincide con la configuración`** — la colección se
  creó con otra dimensión/métrica; exporta `AURUM_ALLOW_RESET=true` y repite
  `make ingest` para reconstruirla desde cero.
- **Puerto 6333 ocupado** — cambia el mapeo en `deploy/qdrant/compose.yaml` y
  `QDRANT_URL` en `.env`.
- **PDF del informe falla** — `make informe` compila LaTeX con
  [Tectonic](https://tectonic-typesetting.github.io/): si no está en el PATH,
  el script descarga el binario (una vez, a `.artifacts/bin/`) y la primera
  compilación baja los paquetes LaTeX necesarios, así que requiere red. El
  diagrama SVG se convierte con CairoSVG, que usa la librería de sistema Cairo
  (presente por defecto en la mayoría de distribuciones).

## 9. Seguridad operativa

- Toda operación valida que la colección empiece por el prefijo
  `aurum-market-eval`; cualquier otro nombre se rechaza antes de crear el cliente.
- La limpieza está desactivada por defecto: recrear una colección exige
  `AURUM_ALLOW_RESET=true` y borrarla exige la confirmación exacta
  `AURUM_CONFIRM_CLEANUP=DELETE:<coleccion>`.
- El repositorio no contiene credenciales; `.env` está en `.gitignore` y la
  plantilla `.env.example` no lleva secretos.
- No hay servicios cloud en el recorrido evaluado: el corrector no hereda costes.
