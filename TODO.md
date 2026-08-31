# Trabajo pendiente

Estado al cierre del 31 de agosto de 2026.

## Hecho: promoción decidida por validación ampliada

La decisión pendiente (¿promover un retador?) se resolvió construyendo una
**validación ampliada** sobre el ESCI público (`make validate-challengers`,
`scripts/validate_challengers.py`): 413 consultas con ≥5 productos juzgados
en catálogo (nivel de decisión) y 1.986 con ≥3 (robustez), comparación
pareada con bootstrap y permutación por signos contra `e5_base_title`.

Resultado — el conjunto grande invirtió el veredicto de las 8 de desarrollo:

- **`e5_large_title` promovida a configuración final**: única cuya mejora
  sobre e5-base sobrevive la corrección de Holm en nDCG y recall con n=413
  (p corregido 0.002; el MRR queda en p=0.040 nominal); con n=1986 las tres
  métricas sobreviven Holm y gana los pareados directos (persistidos en el
  artefacto) a bge-m3 y Qwen3 en las tres.
- El dominio aparente de bge-m3/Qwen3 en desarrollo era mayormente ruido
  (bge-m3 vs e5-base con n=413: p=0.13/0.13/0.51).
- Cascada completa ejecutada: reingesta 1024d + idempotencia, umbral de
  duplicados recalibrado por max-margin (0.943, hueco 0.8978–0.9884),
  evaluate/events/sweep/search-results/evidence, notebooks 00–05 e informe.

## Notas abiertas

- **Probable falso negativo en el ciego**: `EVAL-DUP-004` puntúa 0.9428 —
  dentro del hueco de desarrollo y a 0.0002 del umbral — y queda negativo
  (6 positivos de 14, antes 7). Por la convención de identificadores es casi
  seguro un duplicado real (recall ciego estimado 6/7). Narrado con esa
  franqueza en notebook 04 e informe como argumento de la banda de
  abstención; no se tocó el umbral para cazarlo porque sería calibrar con el
  ciego.
- Con `GEMINI_API_KEY` en `.env`: `make embeddings && make experiments`
  añade la fila de Gemini Embedding 2 a la comparativa (y podría pasarse
  también por `make validate-challengers` con soporte extra).
- El parquet de la validación (51 MB) vive en `data/validacion/` (no
  versionado); la URL la imprime el script si falta.
