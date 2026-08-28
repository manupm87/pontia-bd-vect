# Trabajo pendiente

Estado al cierre del 29 de agosto de 2026 (commit `5174f83`).

## Decisión pendiente: ¿promover un retador a configuración final?

Los retadores locales ya están **medidos** sobre las 8 consultas de desarrollo
(oráculo exacto, composición título+marca+color; ver
`.artifacts/experimentos/registro_experimentos.json`, regenerable con
`make experiments`):

| Configuración | nDCG@10 | Recall@10 | MRR@10 | Dim. |
|---|---|---|---|---|
| `qwen3_embedding_title` | **0.648** | **0.296** | 0.750 | 1024 |
| `bge_m3_title` | 0.637 | 0.266 | **0.875** | 1024 |
| `e5_base_title` *(final actual)* | 0.563 | 0.248 | 0.792 | 768 |
| `e5_large_title` | 0.527 | 0.195 | 0.719 | 1024 |

Hallazgos: bge-m3 y Qwen3 superan con claridad a toda la familia E5; e5-large
rinde *peor* que e5-base (la capacidad no escala monotónicamente en este
dominio). Trade-off abierto: Qwen3 gana en nDCG/recall, bge-m3 en MRR; ambos
son 1024d con encoder más pesado (latencia de la CLI y memoria del índice).

## Si se decide promover (cascada completa, ya hecha una vez en `13cf8b8`)

1. `config/run_config.yaml` → `embedding.configuration`.
2. `AURUM_ALLOW_RESET=true make ingest` (cambia la dimensión) y una segunda
   pasada de `make ingest` como prueba de idempotencia.
3. `make duplicates` → mirar el hueco en `.artifacts/duplicados/calibracion.json`
   y fijar el umbral **max-margin** (punto medio) en `run_config.yaml`; repetir
   `make duplicates`.
4. `make evaluate && AURUM_ALLOW_RESET=true make events` y
   `AURUM_ALLOW_RESET=true make sweep-ef` (la colección débil también cambia de
   dimensión) y `make search-results && make evidence`.
5. Actualizar las cifras narradas en `scripts/notebooks_src/` (00 intro
   dimensión, 01 tablas/decisión, 02 esquema y lectura de latencia, 04 rangos y
   umbral de duplicados, 05 métricas/barrido/atribución) y en
   `docs/informe/INFORME_AURUM_MARKET.tex`; después
   `make execute-notebook` (con `--in-place` vía script) y `make informe`.
6. `make verify && make test-integration`, revisión adversarial, commit.

## Otras ideas anotadas

- Con `GEMINI_API_KEY` en `.env`: `make embeddings && make experiments` añade
  la fila de Gemini Embedding 2 a la comparativa.
- Si se promueve un modelo 1024d, re-medir la latencia de la CLI (el encoder
  domina el tiempo de una consulta interactiva).
