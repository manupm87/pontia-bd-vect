# Referencias primarias y documentación oficial

## Datos y embeddings

- [Shopping Queries Dataset](https://github.com/amazon-science/esci-data): repositorio oficial, esquema y licencia Apache-2.0.
- [ESCI: Improving Product Search with Customer Behavior](https://arxiv.org/abs/2206.06588): publicación original de las etiquetas Exact, Substitute, Complement e Irrelevant.
- [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small): model card, prefijos `query:` y `passage:`, dimensión y normalización.

## FAISS

- [Repositorio oficial de FAISS](https://github.com/facebookresearch/faiss): código, releases e instalación.
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki): documentación de arquitectura, CPU, GPU y benchmarks.
- [Getting started](https://github.com/facebookresearch/faiss/wiki/Getting-started): contrato básico de `train`, `add` y `search`.
- [Faiss indexes](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes): índices Flat, IVF, HNSW, PQ, memoria y operaciones soportadas.
- [Guidelines to choose an index](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index): árbol de decisión y compromisos.
- [MetricType and distances](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances): inner product, L2 y equivalencia con coseno normalizado.
- [Implementation notes](https://github.com/facebookresearch/faiss/wiki/Implementation-notes): estadísticas internas de distancias e implementación.
- [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734): artículo de referencia de FAISS.

## Algoritmos ANN

- [Efficient and Robust Approximate Nearest Neighbor Search Using HNSW](https://arxiv.org/abs/1603.09320): publicación original de HNSW.
- [Product Quantization for Nearest Neighbor Search](https://doi.org/10.1109/TPAMI.2010.57): publicación original de Product Quantization.
- [Product Quantization for Nearest Neighbor Search · versión abierta](https://hal.inria.fr/inria-00514462): manuscrito y metadatos de INRIA.
- [The Inverted Multi-Index](https://ieeexplore.ieee.org/document/6248028): extensión de índices invertidos para espacios vectoriales.

## Interpretación de benchmarks

Los resultados del notebook no son cifras universales de FAISS. Deben registrarse junto a versión, CPU, instrucciones disponibles, threads, batch, dimensión, distribución de datos y parámetros. Recall@k usa Flat como ground truth del índice; no sustituye juicios de relevancia.
