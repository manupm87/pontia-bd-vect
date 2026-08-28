"""Build the student-facing notebook on semantic retrieval and FAISS ANN indexes."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "sesion_02_faiss_indices_ann.ipynb"


def markdown(source: str) -> nbformat.NotebookNode:
    """Create a normalized Markdown cell."""
    return nbformat.v4.new_markdown_cell(dedent(source).strip() + "\n")


def code(source: str) -> nbformat.NotebookNode:
    """Create a normalized code cell."""
    return nbformat.v4.new_code_cell(dedent(source).strip() + "\n")


def build_cells() -> list[nbformat.NotebookNode]:
    """Return the complete linear notebook."""
    cells: list[nbformat.NotebookNode] = []

    cells.extend(
        [
            markdown(
                r"""
                # Sesión práctica 2 · Búsqueda semántica a escala con FAISS

                ## Del embedding al producto que termina apareciendo en pantalla

                Un encoder transforma una consulta y cada producto en vectores. Eso no resuelve por sí solo la búsqueda. Todavía hace falta localizar los $k$ productos más próximos, conservar la relación entre posiciones vectoriales e identificadores de negocio, recuperar los metadatos, aplicar restricciones y ordenar la respuesta final.

                En un catálogo pequeño se puede comparar la consulta con todos los vectores. Esa búsqueda exacta es sencilla y produce el vecino real según la métrica elegida. Su coste, sin embargo, crece con el número de productos $N$, la dimensión $d$ y el número de consultas $Q$. Para un lote de consultas, la operación dominante puede expresarse como una multiplicación $QX^\top$ de coste aproximado $O(QNd)$.

                Los algoritmos **Approximate Nearest Neighbor** reducen el trabajo evitando comparar contra todo el catálogo. A cambio, pueden omitir vecinos que el cálculo exacto habría devuelto. La palabra *aproximado* no se refiere al embedding ni a que el score tenga pocos decimales: se refiere a que el algoritmo de recuperación renuncia a la garantía de encontrar siempre el top-k exacto.

                El marketplace utilizado aquí contiene 50.000 productos españoles reales del Shopping Queries Dataset de Amazon Science. Las consultas incluyen intenciones literales, paráfrasis por necesidad y 256 títulos de producto utilizados como sondas. El objetivo no es declarar un ganador universal, sino medir tres recursos en conflicto: fidelidad del ranking, latencia y memoria.
                """
            ),
            markdown(
                r"""
                ## Índice de contenidos

                1. [Del embedding al sistema de recuperación](#1-del-embedding-al-sistema-de-recuperación)
                2. [Búsqueda exacta](#2-búsqueda-exacta)
                3. [Qué significa buscar aproximadamente](#3-qué-significa-buscar-aproximadamente)
                4. [IVF](#4-ivf-buscar-primero-la-región-del-espacio)
                5. [HNSW](#5-hnsw-navegar-por-un-grafo-de-proximidad)
                6. [Product Quantization](#6-product-quantization-comprimir-para-poder-buscar)
                7. [Comparación de configuraciones](#7-comparar-configuraciones-la-frontera-de-pareto)
                8. [Metadatos, filtros y persistencia](#8-metadatos-filtros-y-persistencia)
                9. [Decisión para el marketplace](#9-decisión-para-el-marketplace)
                """
            ),
            markdown(
                r"""
                ## 1. Del embedding al sistema de recuperación

                ### 1.1. Dos evaluaciones distintas que no deben mezclarse

                La calidad de un buscador semántico tiene al menos dos capas:

                **Calidad del modelo.** Un embedding puede colocar juntos productos irrelevantes. Esa capa se evalúa con juicios de negocio como Exact, Substitute, Complement e Irrelevant y métricas como nDCG. Un índice exacto no arregla una mala geometría: encontrará con precisión los vecinos equivocados del modelo.

                **Fidelidad del índice.** Dado un espacio vectorial fijo, un ANN puede no devolver algunos vecinos que devolvería la fuerza bruta. Esa capa se evalúa comparando el top-k aproximado contra el top-k exacto mediante recall@k.

                $$
                \operatorname{recall@k}(q)=
                \frac{|ANN_k(q)\cap Exact_k(q)|}{k}
                $$

                Un recall@10 de 0,9 significa que, en promedio, nueve de los diez IDs exactos aparecen en el top diez aproximado. No significa que el buscador tenga un 90 % de relevancia ni que el 90 % de los usuarios compre. El recall del índice utiliza el ranking exacto como oráculo técnico; nDCG utiliza juicios humanos como referencia de negocio.

                Esta separación permite atribuir los errores. Si `IndexFlatIP` ya devuelve malos productos, el problema está en representación, modelo o datos. Si Flat funciona y un IVF con `nprobe=1` pierde resultados, el problema se ha introducido en la aproximación.
                """
            ),
            code(
                """
                from pathlib import Path
                import os
                import sys

                current_directory = Path.cwd().resolve()
                project_root = next(
                    candidate
                    for candidate in (current_directory, *current_directory.parents)
                    if (candidate / "pyproject.toml").exists()
                )
                sys.path.insert(0, str(project_root / "src"))
                """
            ),
            code(
                """
                from time import perf_counter

                import faiss
                import numpy as np
                import pandas as pd
                import plotly.express as px
                import plotly.graph_objects as go
                import plotly.io as pio
                from dotenv import load_dotenv

                load_dotenv(project_root / ".env", override=False)
                pio.templates.default = "plotly_white"
                """
            ),
            code(
                """
                from vector_index_session import (
                    benchmark_search,
                    build_flat_index,
                    build_hnsw_index,
                    build_ivf_flat_index,
                    build_ivf_pq_index,
                    configure_faiss_threads,
                    load_session_data,
                    recall_per_query,
                    serialized_size_bytes,
                )

                faiss_threads = configure_faiss_threads()
                session_data = load_session_data(memory_map=True)
                """
            ),
            code(
                """
                products = session_data.products
                queries = session_data.queries
                judgments = session_data.judgments
                product_embeddings = session_data.product_embeddings
                query_embeddings = session_data.query_embeddings

                print(f"Productos: {len(products):,}")
                print(f"Consultas de benchmark: {len(queries):,}")
                print(f"Dimensión: {product_embeddings.shape[1]:,}")
                print(f"Threads FAISS: {faiss_threads}")
                """
            ),
            markdown(
                r"""
                ### 1.2. El flujo completo de una consulta

                El índice vectorial ocupa una posición concreta dentro del sistema. No tokeniza texto, no llama al encoder y no conoce precios, marcas ni stock salvo que otra capa se los proporcione. FAISS recibe matrices numéricas `float32` y devuelve scores e IDs enteros.

                Durante la indexación se prepara el texto del producto, se calcula su embedding, se normaliza si la métrica lo requiere y se añade al índice. En paralelo se guarda una tabla que traduce el ID entero interno a `product_id` y metadatos.

                Durante la consulta se reproduce exactamente el contrato del encoder, se busca el top-k, se hidratan los IDs con la tabla de productos y se aplican reglas posteriores: disponibilidad, permisos, filtros, diversidad o reranking. Si se cambia el modelo, la dimensión, la normalización o la plantilla de entrada, el índice deja de ser compatible aunque la API de FAISS siga aceptando el array.
                """
            ),
            markdown(
                r"""
                ### 1.3. El ciclo de vida `train → add → search`

                Todos los índices de FAISS comparten una interfaz, pero no todos atraviesan las mismas fases. `IndexFlatIP` nace entrenado porque no tiene parámetros estadísticos que aprender. Se crea con la dimensión, se añaden vectores y puede buscar inmediatamente.

                IVF y PQ necesitan `train`. Durante esta fase aprenden centroides y codebooks utilizando una muestra representativa. `is_trained` cambia a `True`, pero el índice todavía no contiene productos: `ntotal` sigue en cero. `add` asigna y codifica cada vector usando los parámetros ya aprendidos. Confundir entrenamiento con indexación produce un artefacto válido pero vacío.

                HNSW no ejecuta un `train` estadístico separado. El trabajo ocurre durante `add`, cuando cada nuevo vector navega el grafo existente, selecciona vecinos y crea enlaces. Por eso `is_trained` no revela el coste real de construcción de un grafo.

                `search` exige consultas `float32` contiguas con la misma dimensión. FAISS no comprueba que procedan del mismo encoder ni que utilicen la misma normalización. Dos matrices de 384 columnas pueden ser incompatibles semánticamente y, aun así, producir scores sin ningún error de ejecución.
                """
            ),
            code(
                """
                pipeline_nodes = [
                    "Consulta", "Encoder", "Vector consulta", "FAISS",
                    "IDs + scores", "Metadatos", "Filtros / reranking", "Resultados",
                ]
                pipeline_figure = go.Figure(
                    go.Sankey(
                        node={"label": pipeline_nodes, "pad": 22, "thickness": 18},
                        link={
                            "source": list(range(7)),
                            "target": list(range(1, 8)),
                            "value": [1] * 7,
                        },
                    )
                )
                pipeline_figure.update_layout(
                    title="FAISS es una etapa del pipeline, no el pipeline completo",
                    height=430,
                )
                pipeline_figure.show()
                """
            ),
            markdown(
                r"""
                ### 1.4. Alineación entre filas, IDs y vectores

                FAISS asigna por defecto IDs consecutivos según el orden de inserción: el primer vector recibe 0, el segundo 1 y así sucesivamente. Los `product_id` del marketplace son strings y no pueden utilizarse directamente como IDs nativos. La tabla `products` conserva ambos mundos: `vector_id` para FAISS y `product_id` para negocio.

                La alineación es un invariante. Si se ordena el DataFrame después de generar la matriz sin aplicar la misma permutación a los vectores, el índice seguirá devolviendo vecinos matemáticamente correctos, pero se mostrarán productos ajenos. Es uno de los fallos más peligrosos porque no produce una excepción.

                El cargador valida que `vector_id` sea exactamente `0..N-1`, que las matrices tengan las mismas filas que sus metadatos, que consulta y documentos compartan dimensión y que todas las normas sean aproximadamente uno. Estas comprobaciones deben ocurrir antes de construir cualquier índice.
                """
            ),
            code(
                """
                products[["vector_id", "product_id", "product_title"]].head()
                """
            ),
            code(
                """
                product_norms = np.linalg.norm(product_embeddings, axis=1)
                query_norms = np.linalg.norm(query_embeddings, axis=1)

                print(f"Normas producto: {product_norms.min():.6f} - {product_norms.max():.6f}")
                print(f"Normas consulta: {query_norms.min():.6f} - {query_norms.max():.6f}")
                """
            ),
        ]
    )

    cells.extend(
        [
            markdown(
                r"""
                # 2. Búsqueda exacta

                ## 2.1. k-NN exacto: el oráculo que necesitamos antes de aproximar

                Con embeddings normalizados, la similitud coseno coincide con el producto escalar. `IndexFlatIP` almacena los vectores completos en `float32` y, al buscar, calcula el producto interno contra todos. No requiere entrenamiento y no modifica la representación.

                El nombre *Flat* no significa que no exista ninguna optimización. FAISS utiliza implementaciones vectorizadas, selección eficiente del top-k y, según la plataforma, instrucciones SIMD o GPU. Lo que no existe es poda algorítmica: todos los candidatos participan en la búsqueda. Por eso garantiza el resultado exacto según los floats almacenados.

                Flat cumple dos papeles. Puede ser el índice de producción si el catálogo y el SLA lo permiten. Además, proporciona el ground truth contra el que se calcula el recall de los ANN. Evaluar un índice aproximado sin un oráculo exacto equivale a medir velocidad sin saber qué resultados se han perdido.
                """
            ),
            code(
                """
                flat_build_started = perf_counter()
                flat_index = build_flat_index(product_embeddings)
                flat_build_seconds = perf_counter() - flat_build_started

                print(f"Entrenado: {flat_index.is_trained}")
                print(f"Vectores: {flat_index.ntotal:,}")
                print(f"Construcción: {flat_build_seconds:.3f} s")
                """
            ),
            code(
                """
                neighbor_count = 10
                exact_scores, exact_ids = flat_index.search(
                    np.ascontiguousarray(query_embeddings),
                    neighbor_count,
                )
                print(exact_scores.shape, exact_ids.shape)
                """
            ),
            markdown(
                r"""
                `search` devuelve dos matrices de forma `(n_queries, k)`. Los scores están ordenados de mayor a menor para inner product. Los IDs ocupan la misma posición que su score. Si el índice contiene menos de $k$ elementos, FAISS completa IDs ausentes con `-1`; ese valor nunca debe utilizarse para indexar un DataFrame porque en pandas o NumPy podría seleccionar accidentalmente la última fila.

                La siguiente función hidrata una consulta concreta. Añade rango y score sin alterar el orden de FAISS. La consulta elegida expresa una necesidad con cambio de unidad: `busco un televisor pequeño de unos setenta centímetros para la cocina`.
                """
            ),
            code(
                """
                def hydrate_ranking(
                    result_ids: np.ndarray,
                    result_scores: np.ndarray,
                ) -> pd.DataFrame:
                    valid_mask = result_ids >= 0
                    valid_ids = result_ids[valid_mask]
                    ranking = products.iloc[valid_ids][
                        ["vector_id", "product_id", "product_title", "product_brand"]
                    ].copy()
                    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
                    ranking["score"] = result_scores[valid_mask]
                    return ranking
                """
            ),
            code(
                """
                demonstration_position = queries.index[
                    queries["workload_id"] == "semantic-101352"
                ].item()
                demonstration_query = queries.iloc[demonstration_position]
                print(demonstration_query["query_text"])
                """
            ),
            code(
                """
                exact_demonstration = hydrate_ranking(
                    exact_ids[demonstration_position],
                    exact_scores[demonstration_position],
                )
                exact_demonstration
                """
            ),
            markdown(
                r"""
                ## 2.2. Verificar FAISS contra la definición matemática

                Como todos los vectores son unitarios, una consulta puede resolverse manualmente mediante `product_embeddings @ query_vector`. Ordenar los scores y tomar los primeros $k$ debe devolver los mismos IDs que `IndexFlatIP`.

                Esta prueba es pequeña pero importante. Confirma conjuntamente la métrica, la normalización y la alineación. Si se utilizara `IndexFlatL2` sobre los mismos vectores unitarios, el ranking también coincidiría porque $\lVert q-x\rVert^2=2-2q^\top x$. Los scores, sin embargo, tendrían otra escala y menor sería mejor.
                """
            ),
            markdown(
                r"""
                ### Calcular scores no es lo mismo que seleccionar top-k

                La multiplicación exhaustiva produce $N$ scores por consulta, pero la aplicación solo necesita los $k$ mejores. Ordenar completamente $N$ elementos costaría $O(N\log N)$. Las implementaciones eficientes utilizan selección parcial, heaps o kernels fusionados para evitar ordenar la cola irrelevante. Cuando $k$ crece, la selección también se encarece aunque el número de productos comparados no cambie.

                El batch modifica el perfil de rendimiento. Una consulta aislada minimiza trabajo total pero aprovecha peor operaciones matriciales y paralelismo. Un lote grande obtiene más throughput, aunque aumenta el tiempo hasta completar todo el batch y puede competir por caché. Por eso `consultas por segundo` y `latencia de una consulta` no son intercambiables.

                En servicios online suele aplicarse micro-batching con una ventana muy corta. La capacidad mejora si llegan consultas suficientes, pero la espera de agrupación pasa a formar parte de la latencia extremo a extremo. El benchmark de este notebook usa un lote fijo para comparar estructuras bajo el mismo patrón, no para decidir por sí solo la estrategia de serving.
                """
            ),
            code(
                """
                manual_scores = product_embeddings @ query_embeddings[demonstration_position]
                manual_ids = np.argsort(-manual_scores, kind="stable")[:neighbor_count]

                np.testing.assert_array_equal(
                    manual_ids,
                    exact_ids[demonstration_position],
                )
                print("El ranking Flat coincide con el producto matricial exhaustivo.")
                """
            ),
            markdown(
                r"""
                ## 2.3. El coste crece con el catálogo

                La complejidad asintótica no sustituye una medición. El tiempo real depende de ancho de banda de memoria, caché, SIMD, número de threads, tamaño del batch y selección top-k. Para aislar el crecimiento con $N$, se fija la dimensión, el hardware, los threads y un lote de 32 consultas.

                El benchmark hace un calentamiento, repite la búsqueda y usa la mediana. No incluye el tiempo del encoder, carga desde disco ni hidratación de metadatos. Por tanto, mide exclusivamente la etapa FAISS y no debe presentarse como latencia extremo a extremo del buscador.
                """
            ),
            code(
                """
                def median_search_ms(
                    index: faiss.Index,
                    query_matrix: np.ndarray,
                    neighbor_limit: int,
                    repeats: int = 8,
                ) -> float:
                    index.search(query_matrix[:4], neighbor_limit)
                    samples = []
                    for _ in range(repeats):
                        started_at = perf_counter()
                        index.search(query_matrix, neighbor_limit)
                        samples.append((perf_counter() - started_at) * 1_000)
                    return float(np.median(samples))
                """
            ),
            code(
                """
                scaling_queries = np.ascontiguousarray(query_embeddings[-32:])
                scaling_rows = []
                for catalog_size in [1_000, 5_000, 10_000, 25_000, 50_000]:
                    sample_index = build_flat_index(product_embeddings[:catalog_size])
                    latency_ms = median_search_ms(
                        sample_index, scaling_queries, neighbor_count
                    )
                    scaling_rows.append(
                        {
                            "products": catalog_size,
                            "latency_ms": latency_ms,
                            "microseconds_per_query": latency_ms * 1_000 / len(scaling_queries),
                        }
                    )
                """
            ),
            code(
                """
                scaling_frame = pd.DataFrame(scaling_rows)
                scaling_figure = px.line(
                    scaling_frame,
                    x="products",
                    y="latency_ms",
                    markers=True,
                )
                scaling_figure.update_layout(
                    title="IndexFlatIP: coste medido al ampliar el catálogo",
                    xaxis_title="Productos indexados",
                    yaxis_title="Mediana de latencia por lote de 32 consultas (ms)",
                )
                scaling_figure.show()
                """
            ),
            markdown(
                r"""
                Si Flat cumple el SLA con margen, su exactitud, simplicidad y facilidad de actualización son ventajas reales. Introducir ANN en 50.000 productos solo por utilizar una tecnología más sofisticada añadiría entrenamiento, parámetros y modos de fallo.

                La necesidad cambia con millones de productos, más consultas concurrentes, dimensiones mayores o presupuestos de CPU estrictos. Para estudiar esa transición sin inventar relevancia, se mantendrá el catálogo real y se observará cuántas comparaciones evita cada estructura.
                """
            ),
        ]
    )

    cells.extend(
        [
            markdown(
                r"""
                # 3. Qué significa buscar aproximadamente

                Un algoritmo ANN construye una estructura que concentra la búsqueda en una fracción prometedora del espacio. La ganancia aparece si esa fracción es mucho menor que $N$ y puede localizarse sin gastar lo mismo que el barrido completo. La pérdida aparece cuando el verdadero vecino queda fuera de la región explorada o su distancia se estima mediante una representación comprimida.

                Las familias principales siguen estrategias diferentes:

                - **Partición del espacio.** IVF agrupa vectores alrededor de centroides y examina algunas listas invertidas.
                - **Grafos de proximidad.** HNSW navega por enlaces entre vecinos desde capas globales hasta una capa densa.
                - **Cuantización.** PQ sustituye floats por códigos compactos y aproxima las distancias mediante tablas.
                - **Proyecciones o árboles.** LSH, random projection forests y variantes de k-d trees dividen el espacio mediante hashes o hiperplanos. Su eficacia depende mucho de dimensión y distribución.

                FAISS no es un algoritmo único. Es una biblioteca que implementa y compone muchas de estas ideas en CPU y GPU. `IndexIVFPQ`, por ejemplo, combina partición IVF y compresión PQ. La cadena de componentes determina dónde aparece la aproximación.
                """
            ),
            markdown(
                r"""
                ## 3.1. Por qué los árboles espaciales pierden fuerza en alta dimensión

                En dos o tres dimensiones, un k-d tree puede descartar grandes regiones comparando la consulta con planos de corte. En cientos de dimensiones, las regiones se solapan respecto a la vecindad buscada y muchas ramas no pueden podarse con seguridad. El recorrido termina visitando una parte sustancial del árbol y se acerca al coste exhaustivo.

                Locality-Sensitive Hashing adopta otra estrategia: diseña funciones hash para que puntos cercanos colisionen con mayor probabilidad. Varias tablas aumentan la oportunidad de colisión, a cambio de memoria y candidatos adicionales. Random projection forests construyen particiones mediante hiperplanos aleatorios. Estas técnicas siguen siendo útiles en ciertos dominios, pero no existe una estructura universal que domine todas las dimensiones, métricas y distribuciones.

                IVF aprovecha clustering aprendido del propio corpus; HNSW aprovecha conectividad local; PQ aprovecha redundancia para comprimir. La popularidad de estas familias en embeddings densos no elimina la necesidad de benchmark. Datos muy agrupados, distribuciones anisotrópicas o cambios de modelo pueden alterar por completo su frontera recall-latencia.
                """
            ),
            markdown(
                r"""
                ## 3.2. Un benchmark ANN necesita un protocolo

                Se fijará `k=10` y se utilizarán los 276 vectores de consulta. `exact_ids` contiene el top-10 de Flat. Para cada configuración se registrará:

                - recall@10 macro;
                - mediana y percentil 95 de latencia por batch;
                - consultas por segundo calculadas sobre la mediana;
                - tamaño serializado del índice;
                - número medio de distancias evaluadas cuando FAISS expone la estadística.

                El paralelismo queda fijado porque comparar un índice con un thread contra otro con ocho confunde algoritmo y recursos. El benchmark hace calentamiento para reducir el efecto de carga perezosa y caché fría. Las medidas siguen siendo locales: sirven para comparar configuraciones en esta máquina, no para prometer un SLA en otra.

                También se separa construcción de consulta. IVF y PQ necesitan entrenamiento; HNSW construye enlaces costosos al insertar; Flat apenas copia memoria. Un índice de baja latencia puede ser inadecuado si tarda demasiado en reconstruirse ante un catálogo que cambia cada hora.
                """
            ),
            code(
                """
                benchmark_records = []
                build_records = []

                flat_size_bytes = serialized_size_bytes(flat_index)
                flat_result, _ = benchmark_search(
                    flat_index,
                    query_embeddings,
                    exact_ids,
                    k=neighbor_count,
                    index_name="Flat exacto",
                    search_parameter="none",
                    parameter_value=0,
                    measured_index_size_bytes=flat_size_bytes,
                )
                benchmark_records.append(flat_result.as_record())
                build_records.append(
                    {"index": "Flat exacto", "build_seconds": flat_build_seconds}
                )
                """
            ),
        ]
    )

    cells.extend(
        [
            markdown(
                r"""
                # 4. IVF: buscar primero la región del espacio

                **Inverted File Index** aprende $nlist$ centroides mediante k-means. Cada producto se asigna al centroide más cercano y su ID se guarda en la lista invertida de esa celda. Al consultar, se comparan primero los centroides, se eligen las `nprobe` celdas más prometedoras y solo se escanean sus productos.

                El entrenamiento minimiza la suma de distancias de cada vector a su centroide asignado. No aprende relevancia ni utiliza las etiquetas ESCI; aproxima la distribución geométrica del catálogo. Si la muestra de entrenamiento no representa productos futuros, las listas pueden quedar desequilibradas y la poda pierde eficacia.

                `IndexIVFFlat` almacena los vectores completos dentro de las listas. La única aproximación es **qué listas se visitan**. Si `nprobe=nlist`, se inspeccionan todas las listas y el ranking debe converger al exacto, aunque con sobrecoste respecto a Flat por la estructura adicional.

                `nlist` controla la granularidad. Demasiadas pocas listas dejan grandes barridos internos; demasiadas listas aumentan el coste de centroides, necesitan más datos de entrenamiento y pueden crear celdas diminutas. `nprobe` es el mando de consulta: aumentarlo mejora recall y aumenta trabajo sin reconstruir el índice.
                """
            ),
            markdown(
                r"""
                ## 4.1. Entrenamiento, asignación y deriva de las listas

                El k-means alterna dos pasos. En la asignación, cada vector de entrenamiento elige su centroide más cercano. En la actualización, cada centroide se mueve a la media de sus asignados. El proceso se repite hasta alcanzar el número de iteraciones o estabilizarse. La inicialización y la muestra influyen, por lo que se fija una semilla.

                Una vez entrenado, `add` calcula el centroide de cada producto y lo inserta en su lista. Los productos nuevos pueden añadirse sin reentrenar, pero los centroides no se desplazan. Si el marketplace incorpora una categoría nueva o cambia el encoder, la distribución puede alejarse de la muestra original. A ese desajuste se le puede llamar deriva del índice.

                La deriva se observa mediante tamaños de lista, distancias a centroides, recall contra un snapshot Flat reciente y distribución de consultas por lista. Reentrenar implica construir otra versión y reasignar todos los productos; modificar centroides dentro del mismo artefacto invalidaría las asignaciones existentes.

                Las listas vacías o muy pequeñas indican que `nlist` puede ser excesivo, la muestra insuficiente o el entrenamiento inestable. Las listas enormes concentran coste y pueden provocar colas de latencia para consultas que caen en regiones densas.
                """
            ),
            code(
                """
                toy_generator = np.random.default_rng(7)
                toy_centers = np.array([[-2, -1], [2, -1], [-1, 2], [2, 2]])
                toy_points = np.vstack(
                    [
                        center + toy_generator.normal(scale=0.55, size=(60, 2))
                        for center in toy_centers
                    ]
                ).astype(np.float32)
                toy_kmeans = faiss.Kmeans(2, 4, niter=30, seed=42, verbose=False)
                toy_kmeans.train(toy_points)
                _, toy_assignments = toy_kmeans.index.search(toy_points, 1)
                """
            ),
            code(
                """
                toy_frame = pd.DataFrame(
                    {
                        "x": toy_points[:, 0],
                        "y": toy_points[:, 1],
                        "list": toy_assignments.ravel().astype(str),
                    }
                )
                toy_figure = px.scatter(
                    toy_frame,
                    x="x",
                    y="y",
                    color="list",
                    title="IVF divide el espacio en listas alrededor de centroides",
                )
                toy_figure.add_trace(
                    go.Scatter(
                        x=toy_kmeans.centroids[:, 0],
                        y=toy_kmeans.centroids[:, 1],
                        mode="markers",
                        marker={"symbol": "x", "size": 18, "color": "black"},
                        name="centroides",
                    )
                )
                toy_figure.show()
                """
            ),
            markdown(
                r"""
                La frontera entre celdas explica un fallo típico. Un vecino real puede quedar en la segunda celda más cercana al query aunque esté muy cerca geométricamente. Con `nprobe=1` nunca será considerado. Aumentar `nprobe` reduce ese error de frontera.

                Para 50.000 vectores se utilizan 256 listas y 30.000 vectores de entrenamiento. Es una configuración de laboratorio, no una fórmula universal. FAISS propone equilibrar el coste de comparar centroides y escanear listas; una regla aproximada sitúa `nlist` alrededor de un múltiplo de $\sqrt{N}$, pero la selección final debe surgir de la curva recall-latencia.
                """
            ),
            code(
                """
                ivf_build_started = perf_counter()
                ivf_index = build_ivf_flat_index(
                    product_embeddings,
                    nlist=256,
                    training_size=30_000,
                    seed=42,
                )
                ivf_build_seconds = perf_counter() - ivf_build_started

                print(f"Entrenado: {ivf_index.is_trained}")
                print(f"Vectores: {ivf_index.ntotal:,}")
                print(f"Construcción: {ivf_build_seconds:.2f} s")
                """
            ),
            code(
                """
                ivf_list_sizes = np.array(
                    [ivf_index.invlists.list_size(list_id) for list_id in range(ivf_index.nlist)]
                )
                pd.Series(ivf_list_sizes).describe(
                    percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
                )
                """
            ),
            code(
                """
                list_figure = px.histogram(
                    x=ivf_list_sizes,
                    nbins=30,
                    labels={"x": "Productos por lista", "y": "Número de listas"},
                    title="El k-means no produce listas perfectamente equilibradas",
                )
                list_figure.show()
                """
            ),
            markdown(
                r"""
                El desequilibrio importa porque `nprobe=8` no implica examinar exactamente $8/256$ del catálogo. Algunas listas contienen más productos y ciertas consultas caen en regiones densas. Las estadísticas internas de FAISS permiten observar el número real de distancias calculadas.

                El índice se barre ahora con varios valores de `nprobe`. La estructura y el tamaño serializado no cambian; solo cambia el presupuesto de exploración de cada consulta.
                """
            ),
            code(
                """
                ivf_size_bytes = serialized_size_bytes(ivf_index)
                ivf_sweep_records = []
                ivf_rankings = {}
                for nprobe in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                    ivf_index.nprobe = nprobe
                    result, result_ids = benchmark_search(
                        ivf_index,
                        query_embeddings,
                        exact_ids,
                        k=neighbor_count,
                        index_name="IVF-Flat",
                        search_parameter="nprobe",
                        parameter_value=nprobe,
                        measured_index_size_bytes=ivf_size_bytes,
                    )
                    ivf_sweep_records.append(result.as_record())
                    ivf_rankings[nprobe] = result_ids
                """
            ),
            code(
                """
                ivf_frame = pd.DataFrame(ivf_sweep_records)
                benchmark_records.extend(ivf_sweep_records)
                build_records.append(
                    {"index": "IVF-Flat", "build_seconds": ivf_build_seconds}
                )
                ivf_frame[
                    [
                        "value",
                        "recall_at_k",
                        "median_latency_ms",
                        "distance_computations_per_query",
                    ]
                ]
                """
            ),
            code(
                """
                ivf_figure = px.line(
                    ivf_frame,
                    x="median_latency_ms",
                    y="recall_at_k",
                    text="value",
                    markers=True,
                )
                ivf_figure.update_traces(textposition="top center")
                ivf_figure.update_layout(
                    title="IVF-Flat: nprobe desplaza el punto recall-latencia",
                    xaxis_title="Mediana de latencia por batch (ms)",
                    yaxis_title="Recall@10 contra Flat",
                )
                ivf_figure.show()
                """
            ),
            markdown(
                r"""
                ## 4.2. Inspeccionar una consulta que IVF pierde

                La media oculta distribuciones. Para `nprobe=1` se calcula recall por consulta y se localiza el peor caso. Comparar sus IDs aproximados y exactos muestra que la pérdida no altera los scores de los candidatos visitados; sencillamente algunos vecinos nunca entraron en el conjunto escaneado.

                Este análisis ayuda a elegir slices de ajuste. Puede que consultas de categorías densas necesiten mayor `nprobe`, mientras otras funcionen con uno. FAISS permite parámetros por consulta en algunas APIs, pero una política adaptativa necesita una señal fiable y complica observabilidad y capacidad.
                """
            ),
            code(
                """
                ivf_low_recall = recall_per_query(
                    exact_ids,
                    ivf_rankings[1],
                    k=neighbor_count,
                )
                worst_ivf_position = int(np.argmin(ivf_low_recall))
                print(queries.iloc[worst_ivf_position]["query_text"])
                print(f"Recall@10: {ivf_low_recall[worst_ivf_position]:.2f}")
                """
            ),
            code(
                """
                pd.DataFrame(
                    {
                        "rank": np.arange(1, neighbor_count + 1),
                        "exact_id": exact_ids[worst_ivf_position],
                        "ivf_nprobe_1_id": ivf_rankings[1][worst_ivf_position],
                        "coincide": np.isin(
                            ivf_rankings[1][worst_ivf_position],
                            exact_ids[worst_ivf_position],
                        ),
                    }
                )
                """
            ),
        ]
    )

    cells.extend(
        [
            markdown(
                r"""
                # 5. HNSW: navegar por un grafo de proximidad

                HNSW construye un grafo donde cada vector mantiene enlaces hacia vecinos. La capa inferior contiene todos los elementos y ofrece conexiones locales densas. Las capas superiores contienen subconjuntos cada vez más pequeños y actúan como autopistas de largo alcance.

                Una búsqueda comienza en un punto de entrada de la capa superior. Se mueve de forma greedy hacia vecinos que mejoran la distancia. Cuando ya no encuentra mejora, desciende de capa y repite con mayor detalle. En la capa cero mantiene una lista de candidatos y explora hasta agotar el presupuesto `efSearch`.

                La jerarquía evita empezar a ciegas en el grafo completo. Su idea recuerda a una skip list: pocas conexiones globales permiten acercarse rápidamente a la región correcta; conexiones locales refinan el resultado.

                La aproximación aparece porque la navegación greedy y el presupuesto finito pueden quedar atrapados en una región. Aumentar conectividad o exploración reduce ese riesgo, pero consume memoria, construcción o latencia.
                """
            ),
            markdown(
                r"""
                ## 5.1. Cómo se construye la jerarquía

                Cada elemento recibe aleatoriamente un nivel máximo con probabilidad decreciente exponencialmente. Todos aparecen en la capa cero; pocos alcanzan las capas superiores. Esa distribución crea una jerarquía sin entrenar centroides globales.

                Al insertar, el algoritmo desciende desde el punto de entrada para encontrar una región cercana. En cada capa relevante explora candidatos y selecciona vecinos. La selección no se limita siempre a los $M$ puntos de menor distancia: una heurística de diversidad evita llenar el vecindario con nodos casi redundantes y conserva enlaces hacia direcciones distintas. Esa conectividad facilita escapar de mínimos locales durante la búsqueda.

                Los enlaces suelen ser bidireccionales y, si un nodo supera su capacidad, su lista se poda. La construcción es incremental; no existe un paso final que optimice globalmente todo el grafo. Por eso `efConstruction` importa: con pocos candidatos, una mala conexión inicial puede persistir y no se corrige simplemente aumentando `efSearch`.

                La altura esperada crece logarítmicamente y las capas superiores permiten saltos largos. Esa intuición no constituye una garantía de latencia para cualquier distribución: la medición sigue siendo necesaria.
                """
            ),
            markdown(
                r"""
                ## 5.2. Los tres controles principales

                `M` limita aproximadamente el número de vecinos almacenados por nodo. Un `M` alto mejora conectividad y recall, especialmente en datos complejos, pero aumenta memoria y coste de inserción.

                `efConstruction` controla cuántos candidatos se exploran al insertar. Un valor alto construye enlaces de mejor calidad a cambio de más tiempo. Es un parámetro de construcción: cambiarlo después no repara el grafo existente.

                `efSearch` controla el ancho de exploración en consulta. Debe ser al menos $k$ y puede ajustarse sin reconstruir. Como `nprobe`, dibuja una curva recall-latencia; no existe un valor correcto independiente del dataset y del SLA.

                `IndexHNSWFlat` conserva los vectores completos, de modo que la distancia de un candidato visitado es exacta. La pérdida procede del recorrido del grafo, no de compresión. Su memoria supera a Flat porque a los $4d$ bytes del vector se añaden enlaces y niveles.
                """
            ),
            code(
                """
                hnsw_build_started = perf_counter()
                hnsw_index = build_hnsw_index(
                    product_embeddings,
                    graph_degree=24,
                    ef_construction=120,
                )
                hnsw_build_seconds = perf_counter() - hnsw_build_started

                print(f"Vectores: {hnsw_index.ntotal:,}")
                print(f"M: {hnsw_index.hnsw.nb_neighbors(0) // 2}")
                print(f"efConstruction: {hnsw_index.hnsw.efConstruction}")
                print(f"Construcción: {hnsw_build_seconds:.2f} s")
                """
            ),
            code(
                """
                hnsw_size_bytes = serialized_size_bytes(hnsw_index)
                hnsw_sweep_records = []
                hnsw_rankings = {}
                for ef_search in [10, 16, 24, 32, 48, 64, 96, 128, 192, 256]:
                    hnsw_index.hnsw.efSearch = ef_search
                    result, result_ids = benchmark_search(
                        hnsw_index,
                        query_embeddings,
                        exact_ids,
                        k=neighbor_count,
                        index_name="HNSW-Flat",
                        search_parameter="efSearch",
                        parameter_value=ef_search,
                        measured_index_size_bytes=hnsw_size_bytes,
                    )
                    hnsw_sweep_records.append(result.as_record())
                    hnsw_rankings[ef_search] = result_ids
                """
            ),
            code(
                """
                hnsw_frame = pd.DataFrame(hnsw_sweep_records)
                benchmark_records.extend(hnsw_sweep_records)
                build_records.append(
                    {"index": "HNSW-Flat", "build_seconds": hnsw_build_seconds}
                )
                hnsw_frame[
                    [
                        "value",
                        "recall_at_k",
                        "median_latency_ms",
                        "distance_computations_per_query",
                    ]
                ]
                """
            ),
            code(
                """
                hnsw_figure = px.line(
                    hnsw_frame,
                    x="median_latency_ms",
                    y="recall_at_k",
                    text="value",
                    markers=True,
                )
                hnsw_figure.update_traces(textposition="top center")
                hnsw_figure.update_layout(
                    title="HNSW: efSearch controla cuánto se navega",
                    xaxis_title="Mediana de latencia por batch (ms)",
                    yaxis_title="Recall@10 contra Flat",
                )
                hnsw_figure.show()
                """
            ),
            markdown(
                r"""
                ## 5.3. Construcción, mutaciones y memoria

                HNSW admite inserciones, pero cada alta debe navegar el grafo y elegir vecinos. El orden de inserción y los parámetros de construcción pueden influir. En la implementación HNSW de FAISS, eliminar elementos no es una operación general soportada porque rompería conexiones; una estrategia habitual marca IDs como inactivos en otra capa y reconstruye periódicamente.

                Para un marketplace con altas y bajas frecuentes, esta limitación forma parte del coste del índice. La latencia de consulta no debe evaluarse aislada del ciclo de vida: carga inicial, actualizaciones, snapshots, despliegue paralelo y rollback.

                El tamaño serializado medirá cuánto cuestan los enlaces frente a Flat e IVF. No equivale exactamente al RSS del proceso, que incluye estructuras temporales, allocator y páginas, pero es una medida reproducible del artefacto persistido.
                """
            ),
        ]
    )

    cells.extend(
        [
            markdown(
                r"""
                # 6. Product Quantization: comprimir para poder buscar

                Flat, IVF-Flat y HNSW-Flat almacenan las 384 coordenadas `float32`: 1.536 bytes por producto antes de IDs y estructura. En millones de productos, la memoria puede convertirse en la restricción principal.

                **Product Quantization** divide el vector en $m$ subvectores. Para cada subespacio aprende un codebook de $2^{nbits}$ centroides. En vez de guardar los floats, cada producto almacena el índice del centroide más cercano en cada subespacio.

                Con `m=48` y `nbits=8`, cada subvector elige uno de 256 centroides y su código ocupa un byte. El producto completo necesita aproximadamente 48 bytes de código en lugar de 1.536 bytes de floats: una compresión teórica de 32 veces antes de codebooks, IDs e IVF.

                En consulta se utiliza **Asymmetric Distance Computation**. La consulta permanece en float; se calcula su distancia a cada centroide de cada subespacio y se construyen tablas. La distancia a un producto se aproxima sumando las entradas indicadas por sus códigos. No hace falta reconstruir todo el vector, pero la distancia deja de ser exacta.
                """
            ),
            markdown(
                r"""
                ## 6.1. Los parámetros `m` y `nbits` cambian capacidad y coste

                Si $d=384$ y $m=48$, cada subvector tiene ocho dimensiones. Un $m$ mayor crea subvectores más cortos y guarda más bytes por producto; suele reducir error porque cada codebook modela un espacio menor. Un $m$ demasiado bajo comprime más, pero obliga a un solo centroide a resumir demasiada información conjunta. FAISS exige que la dimensión sea divisible por $m$ en la configuración utilizada.

                `nbits=8` ofrece 256 centroides por subespacio y permite almacenar cada elección en un byte. Aumentar bits amplía el codebook y puede reducir distorsión, pero necesita más entrenamiento, memoria de tablas y almacenamiento de códigos. Reducir a cuatro bits empaqueta dos elecciones por byte, con una representación más agresiva.

                Para una consulta $q$ dividida en subvectores $q_j$ y un producto codificado mediante centroides $c_{j,a_j}$, ADC aproxima la distancia como

                $$\widetilde{d}(q,x)=\sum_{j=1}^{m}d(q_j,c_{j,a_j}).$$

                La tabla contiene las distancias de cada $q_j$ a todos los centroides del subespacio. Cada producto requiere después lecturas y sumas, no 384 multiplicaciones float. La velocidad final depende también de layout, SIMD y acceso a memoria; menos bytes no garantiza automáticamente menor latencia en cualquier tamaño.
                """
            ),
            markdown(
                r"""
                ## 6.2. IVF-PQ combina dos fuentes de aproximación

                `IndexIVFPQ` primero asigna productos a listas IVF. Dentro de cada lista codifica normalmente el residual entre el vector y su centroide grueso. El residual tiene menos estructura global que el vector original y suele cuantizarse mejor.

                Al buscar pueden perderse vecinos por dos razones:

                1. su lista no se visita porque `nprobe` es bajo;
                2. la distancia cuantizada altera el orden incluso dentro de las listas visitadas.

                Aumentar `nprobe` solo corrige la primera. Si la curva de recall se estanca aunque se visiten muchas listas, el límite procede del código PQ. Se puede aumentar `m`, usar más bits, aplicar una transformación OPQ o rerankear un conjunto mayor con los vectores originales si se conservan en almacenamiento secundario.

                PQ necesita entrenamiento suficiente para cada codebook. Con pocos ejemplos aparecen centroides mal aprendidos o avisos de FAISS. La muestra de entrenamiento debe representar la distribución que se indexará.
                """
            ),
            code(
                """
                pq_build_started = perf_counter()
                ivf_pq_index = build_ivf_pq_index(
                    product_embeddings,
                    nlist=256,
                    subquantizers=48,
                    bits_per_code=8,
                    training_size=30_000,
                    seed=42,
                )
                pq_build_seconds = perf_counter() - pq_build_started

                print(f"Code size: {ivf_pq_index.code_size} bytes por producto")
                print(f"Construcción: {pq_build_seconds:.2f} s")
                """
            ),
            code(
                """
                pq_size_bytes = serialized_size_bytes(ivf_pq_index)
                pq_sweep_records = []
                pq_rankings = {}
                for nprobe in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                    ivf_pq_index.nprobe = nprobe
                    result, result_ids = benchmark_search(
                        ivf_pq_index,
                        query_embeddings,
                        exact_ids,
                        k=neighbor_count,
                        index_name="IVF-PQ",
                        search_parameter="nprobe",
                        parameter_value=nprobe,
                        measured_index_size_bytes=pq_size_bytes,
                    )
                    pq_sweep_records.append(result.as_record())
                    pq_rankings[nprobe] = result_ids
                """
            ),
            code(
                """
                pq_frame = pd.DataFrame(pq_sweep_records)
                benchmark_records.extend(pq_sweep_records)
                build_records.append(
                    {"index": "IVF-PQ", "build_seconds": pq_build_seconds}
                )
                pq_frame[
                    [
                        "value",
                        "recall_at_k",
                        "median_latency_ms",
                        "distance_computations_per_query",
                    ]
                ]
                """
            ),
            code(
                """
                pq_figure = px.line(
                    pq_frame,
                    x="median_latency_ms",
                    y="recall_at_k",
                    text="value",
                    markers=True,
                )
                pq_figure.update_traces(textposition="top center")
                pq_figure.update_layout(
                    title="IVF-PQ: nprobe no elimina el error de cuantización",
                    xaxis_title="Mediana de latencia por batch (ms)",
                    yaxis_title="Recall@10 contra Flat",
                )
                pq_figure.show()
                """
            ),
            markdown(
                r"""
                Si el recall se aplana antes de uno, visitar más listas ya no basta. La distancia aproximada de PQ ha cambiado el orden de algunos candidatos. Esta diferencia muestra por qué `ANN` no es una sola causa de error y por qué un único parámetro no siempre puede recuperar calidad.

                Existen cuantizadores escalares que codifican cada coordenada por separado, variantes PQ aceleradas, OPQ que rota el espacio antes de dividirlo y cuantizadores aditivos. La familia correcta depende del presupuesto de memoria y del recall objetivo. En este material se mantiene una configuración comprensible para que cada pérdida tenga una causa identificable.
                """
            ),
        ]
    )

    cells.extend(
        [
            markdown(
                r"""
                # 7. Comparar configuraciones: la frontera de Pareto

                Una tabla con `el más rápido`, `el de mayor recall` y `el más pequeño` produce tres ganadores distintos. La decisión necesita restricciones. Por ejemplo: recall@10 mínimo de 0,95, memoria disponible y latencia p95 compatible con el SLA.

                Un punto está dominado si existe otro con recall igual o mayor y latencia igual o menor. Los puntos no dominados forman una frontera de Pareto. Cambiar `nprobe` o `efSearch` desplaza una configuración sobre esa frontera; cambiar la estructura altera la forma de la curva y la memoria.

                El gráfico siguiente incluye Flat como referencia exacta y todos los barridos. El tamaño del punto representa el tamaño serializado. Los resultados solo son válidos para este hardware, thread count, batch, datos y versión de FAISS.
                """
            ),
            markdown(
                r"""
                ## 7.1. Recall depende de $k$ y de la distribución de consultas

                Un índice puede obtener recall@1 alto y recall@100 bastante menor. Los primeros vecinos suelen estar más separados y resultan fáciles; al ampliar $k$ entran candidatos con scores parecidos y pequeñas aproximaciones cambian el orden o la pertenencia al conjunto. El valor de $k$ debe coincidir con el número de candidatos que consume la siguiente etapa, no con un estándar arbitrario.

                La media macro da el mismo peso a cada consulta. Conviene acompañarla con percentiles y slices porque los peores casos pueden concentrarse en consultas raras, categorías densas o regiones mal representadas durante entrenamiento. Un promedio de 0,95 puede esconder un grupo con recall cero.

                Las 256 sondas basadas en títulos amplían la cobertura geométrica del benchmark, mientras las consultas de negocio conservan formas literales y paráfrasis. Las sondas no tienen juicios de relevancia y no deben utilizarse para nDCG; su función es ejercitar regiones del espacio y estabilizar la medida de fidelidad ANN.

                En producción el workload debe muestrearse desde logs, respetar privacidad y mantener una partición temporal que detecte deriva. Ajustar y evaluar sobre las mismas consultas puede sobreoptimizar parámetros para un snapshot.
                """
            ),
            code(
                """
                benchmark_frame = pd.DataFrame(benchmark_records)
                benchmark_frame["index_size_mib"] = (
                    benchmark_frame["index_size_bytes"] / 2**20
                )
                benchmark_frame["configuration"] = (
                    benchmark_frame["parameter"]
                    + "="
                    + benchmark_frame["value"].astype(str)
                )
                """
            ),
            code(
                """
                pareto_figure = px.scatter(
                    benchmark_frame,
                    x="median_latency_ms",
                    y="recall_at_k",
                    color="index",
                    size="index_size_mib",
                    hover_data=[
                        "configuration",
                        "p95_latency_ms",
                        "queries_per_second",
                        "distance_computations_per_query",
                    ],
                    size_max=42,
                )
                pareto_figure.update_layout(
                    title="Recall, latencia y memoria de todas las configuraciones",
                    xaxis_title="Mediana de latencia por batch (ms)",
                    yaxis_title="Recall@10 contra Flat",
                    height=620,
                )
                pareto_figure.show()
                """
            ),
            code(
                """
                size_frame = (
                    benchmark_frame.groupby("index", as_index=False)["index_size_mib"]
                    .first()
                    .sort_values("index_size_mib")
                )
                size_figure = px.bar(
                    size_frame,
                    x="index",
                    y="index_size_mib",
                    text_auto=".1f",
                    title="Tamaño serializado de cada estructura",
                )
                size_figure.update_layout(yaxis_title="MiB")
                size_figure.show()
                """
            ),
            code(
                """
                build_frame = pd.DataFrame(build_records)
                build_figure = px.bar(
                    build_frame,
                    x="index",
                    y="build_seconds",
                    text_auto=".2f",
                    title="El coste de construcción también forma parte de la decisión",
                )
                build_figure.update_layout(yaxis_title="Segundos")
                build_figure.show()
                """
            ),
            markdown(
                r"""
                ## 7.2. Seleccionar bajo una restricción explícita

                Se filtran configuraciones con recall@10 al menos 0,95 y se elige la menor mediana de latencia. Este procedimiento no convierte 0,95 en un umbral universal. Hace visible la política. Un buscador médico, una deduplicación offline y un carrusel comercial toleran pérdidas distintas.

                El umbral del índice tampoco garantiza relevancia. Si el vecino exacto número diez es irrelevante, recuperarlo aumenta recall técnico sin ayudar al usuario. Para validar el sistema completo se necesita observar además nDCG, conversión, abandono y slices de negocio.
                """
            ),
            code(
                """
                recall_threshold = 0.95
                eligible_configurations = benchmark_frame.loc[
                    benchmark_frame["recall_at_k"] >= recall_threshold
                ].sort_values("median_latency_ms")
                eligible_configurations[
                    [
                        "index",
                        "configuration",
                        "recall_at_k",
                        "median_latency_ms",
                        "p95_latency_ms",
                        "index_size_mib",
                    ]
                ].head(10)
                """
            ),
        ]
    )

    cells.extend(
        [
            markdown(
                r"""
                # 8. Metadatos, filtros y persistencia

                ## 8.1. FAISS solo conoce vectores e IDs

                Un marketplace rara vez busca sin restricciones. Puede exigir stock, país, categoría, permisos, precio o marca. FAISS no almacena esas columnas como una base de datos documental. El patrón más simple recupera más de $k$ candidatos, hidrata metadatos y filtra después.

                El **post-filter** puede devolver menos de $k$ resultados si pocos candidatos cumplen la condición. Aumentar el *oversampling* ayuda, pero aumenta latencia y no garantiza llenar el cupo para filtros muy selectivos.

                Un **pre-filter** obtiene primero los IDs permitidos y restringe la búsqueda. Puede ser correcto pero perder eficiencia si el ANN no admite el selector de forma nativa o si obliga a construir muchos subíndices. Algunas variantes IVF de FAISS aceptan selectores de IDs; otra estrategia mantiene índices por segmento. La arquitectura depende de cardinalidad, combinación de filtros y frecuencia de actualización.
                """
            ),
            markdown(
                r"""
                ### Elegir entre post-filter, selectores y particiones

                El post-filter mantiene un único índice y es sencillo cuando el filtro elimina pocos candidatos. Su coste se vuelve impredecible con condiciones selectivas: para obtener diez resultados quizá haya que recuperar miles, y el ANN se optimizó para su top-k global, no para el subconjunto permitido.

                Los selectores de ID restringen los candidatos aceptables durante el escaneo de algunas estructuras. Funcionan mejor si la capa de metadatos puede producir rápidamente un bitmap o conjunto de IDs y si FAISS puede aplicar esa restricción sin recorrer listas inútiles. La compatibilidad varía por tipo de índice y API.

                Particionar por país o gran categoría reduce el universo antes de buscar y permite configuraciones distintas. Demasiadas particiones pequeñas empeoran entrenamiento, operación y combinaciones de filtros. También puede duplicar productos que pertenecen a varios segmentos.

                Una base de datos vectorial integra normalmente metadatos, filtros, replicación y durabilidad alrededor de una estructura ANN. FAISS ofrece los algoritmos de búsqueda, pero la aplicación conserva la responsabilidad de coordinar esas capas.
                """
            ),
            code(
                """
                filter_scores, filter_ids = flat_index.search(
                    np.ascontiguousarray(
                        query_embeddings[demonstration_position : demonstration_position + 1]
                    ),
                    500,
                )
                filter_candidates = products.iloc[filter_ids[0]].copy()
                common_brands = filter_candidates["product_brand"].value_counts()
                target_brand = common_brands.index[0]
                print(f"Filtro de ejemplo: product_brand == {target_brand!r}")
                """
            ),
            code(
                """
                filter_rows = []
                for retrieval_depth in [10, 25, 50, 100, 250, 500]:
                    candidate_ids = filter_ids[0, :retrieval_depth]
                    candidate_products = products.iloc[candidate_ids]
                    matching_count = int(
                        candidate_products["product_brand"].eq(target_brand).sum()
                    )
                    filter_rows.append(
                        {
                            "retrieval_depth": retrieval_depth,
                            "matching_results": min(matching_count, 10),
                        }
                    )
                """
            ),
            code(
                """
                filter_frame = pd.DataFrame(filter_rows)
                filter_figure = px.line(
                    filter_frame,
                    x="retrieval_depth",
                    y="matching_results",
                    markers=True,
                    title="Post-filter: recuperar más no siempre llena el top-10",
                )
                filter_figure.update_layout(
                    xaxis_title="Candidatos recuperados antes del filtro",
                    yaxis_title="Resultados que cumplen la marca (máximo 10)",
                )
                filter_figure.show()
                """
            ),
            markdown(
                r"""
                El ejemplo elige una marca presente en la vecindad para visualizar el mecanismo. En un sistema real, el filtro procede de la petición y su selectividad puede variar varios órdenes de magnitud. Debe medirse recall condicionado al filtro, no solo recall global.

                También hay que decidir el orden con reglas de negocio. Filtrar productos sin stock suele ser obligatorio antes de mostrar. Diversificar marcas o aplicar popularidad puede ocurrir después de la recuperación. Cada transformación cambia el ranking visible y necesita observabilidad separada.
                """
            ),
        ]
    )

    cells.extend(
        [
            markdown(
                r"""
                ## 8.2. Persistencia, carga y ciclo de vida

                Un índice construido en memoria no es un despliegue. FAISS puede serializar el artefacto con `write_index` y cargarlo con `read_index`. El fichero debe versionarse junto a modelo, dimensión, métrica, normalización, plantilla de texto, snapshot de datos y parámetros de construcción.

                La tabla de metadatos necesita el mismo snapshot que el índice. Publicar uno sin el otro rompe la traducción de IDs. Una estrategia segura construye una versión nueva en paralelo, ejecuta pruebas de integridad y calidad, cambia un alias o puntero de forma atómica y conserva la versión anterior para rollback.

                `read_index` no debe utilizarse con ficheros no confiables. Un índice es un artefacto binario complejo, no un formato seguro para entradas arbitrarias. También debe verificarse checksum y procedencia antes de cargarlo.
                """
            ),
            code(
                """
                artifact_directory = project_root / ".artifacts" / "indexes"
                artifact_directory.mkdir(parents=True, exist_ok=True)
                hnsw_path = artifact_directory / "hnsw_flat.faiss"

                hnsw_index.hnsw.efSearch = 96
                faiss.write_index(hnsw_index, str(hnsw_path))
                reloaded_hnsw = faiss.read_index(str(hnsw_path))
                reloaded_hnsw.hnsw.efSearch = 96
                """
            ),
            code(
                """
                _, original_hnsw_ids = hnsw_index.search(
                    np.ascontiguousarray(query_embeddings[:8]), neighbor_count
                )
                _, reloaded_hnsw_ids = reloaded_hnsw.search(
                    np.ascontiguousarray(query_embeddings[:8]), neighbor_count
                )
                np.testing.assert_array_equal(original_hnsw_ids, reloaded_hnsw_ids)
                print(f"Índice recargado correctamente desde {hnsw_path.name}")
                """
            ),
            markdown(
                r"""
                # 9. Decisión para el marketplace

                El caso de uso no obliga a utilizar ANN. Con 50.000 productos, Flat puede ser suficientemente rápido y ofrece exactitud, inserciones sencillas y menor complejidad operativa. Si la medición cumple el SLA bajo concurrencia real, esa es una conclusión válida.

                Al crecer el catálogo o la carga, las curvas permiten elegir con evidencia. **IVF-Flat** resulta atractivo cuando se acepta entrenamiento y se quiere controlar claramente la fracción del catálogo escaneada. Conserva vectores completos, pero obliga a vigilar la deriva de centroides y el equilibrio de las listas.

                **HNSW-Flat** suele ofrecer una frontera recall-latencia potente y admite inserciones, a cambio de memoria adicional, una construcción más costosa y una estrategia específica para bajas. **IVF-PQ**, en cambio, responde a una restricción de memoria: su compresión puede ser enorme, pero introduce error de distancia además de la poda de IVF.

                Ninguna elección queda cerrada solo con este notebook. La configuración debe repetirse con volumen objetivo, hardware de producción, concurrencia, filtros reales y distribución de consultas. Lo que sí queda establecido es el método: Flat crea el oráculo; recall cuantifica la aproximación; las curvas muestran el intercambio; el negocio fija la restricción.
                """
            ),
        ]
    )

    return cells


def add_execution_bridges(
    cells: list[nbformat.NotebookNode],
) -> list[nbformat.NotebookNode]:
    """Explain the purpose of every experiment before its first code cell."""

    bridges = {
        "## 1. Del embedding": (
            "Empezaremos cargando el entorno, los 50.000 productos, las consultas y las matrices "
            "de embeddings. Antes de construir un índice comprobaremos formas, tipos y metadatos; "
            "la búsqueda solo será interpretable si esos artefactos comparten el mismo contrato."
        ),
        "### 1.3. El ciclo": (
            "Representaremos este ciclo de vida como una tubería completa, desde el texto de la "
            "query hasta la hidratación de productos. El diagrama nos servirá para ubicar después "
            "qué coste y qué posible fallo pertenece a cada etapa."
        ),
        "### 1.4. Alineación": (
            "Vamos a inspeccionar el mapeo `vector_id → product_id` y la distribución de normas. "
            "Así confirmaremos que las filas de la matriz pueden traducirse a productos y que el "
            "producto escalar implementa realmente el coseno esperado."
        ),
        "# 2. Búsqueda exacta": (
            "Construiremos `IndexFlatIP`, añadiremos todos los vectores y registraremos tiempo, "
            "tamaño y número de elementos. Después buscaremos el top-10 exacto de todo el workload; "
            "esos IDs serán el oráculo común de los índices aproximados."
        ),
        "`search` devuelve": (
            "Crearemos una función de hidratación que descarte el centinela `-1` y una los IDs "
            "devueltos con sus títulos. La aplicaremos a una consulta semántica concreta para leer "
            "el ranking como productos, no como dos matrices anónimas."
        ),
        "### Calcular scores": (
            "Calcularemos manualmente los productos escalares de esa misma query, seleccionaremos "
            "su top-k y exigiremos que coincida con FAISS. La prueba fijará la definición exacta "
            "antes de introducir ninguna aproximación."
        ),
        "## 2.3. El coste": (
            "Mediremos ahora Flat sobre prefijos crecientes del catálogo manteniendo fijas la "
            "dimensión, las consultas y los threads. El gráfico mostrará el escalado observado en "
            "esta máquina, sin convertirlo en una cifra universal."
        ),
        "## 3.2. Un benchmark": (
            "Prepararemos una estructura de registros común para todos los índices y anotaremos la "
            "configuración exacta de Flat. Cada barrido posterior añadirá recall, p50, p95, "
            "throughput, construcción y memoria con el mismo esquema."
        ),
        "## 4.1. Entrenamiento": (
            "Antes de entrenar IVF sobre 384 dimensiones, construiremos una partición bidimensional "
            "de juguete. Visualizaremos centroides, asignaciones y fronteras para hacer visible el "
            "error que después mediremos sobre el catálogo real."
        ),
        "La frontera entre celdas": (
            "Entrenaremos a continuación `IndexIVFFlat` con 256 listas sobre una muestra "
            "determinista y añadiremos los 50.000 productos. Inspeccionaremos el tamaño de las listas "
            "antes de barrer `nprobe`, porque visitar ocho listas no equivale a visitar ocho listas iguales."
        ),
        "El desequilibrio importa": (
            "Ejecutaremos el barrido de `nprobe` manteniendo fijo el índice. Para cada valor "
            "mediremos cuántas distancias calcula FAISS, qué recall conserva y qué latencia paga; "
            "así podremos relacionar la geometría de las listas con la curva resultante."
        ),
        "## 4.2. Inspeccionar": (
            "Localizaremos la query con peor recall cuando `nprobe` es bajo y enfrentaremos sus IDs "
            "exactos con los aproximados. Este caso concreto permitirá explicar la pérdida que la "
            "media del benchmark resume."
        ),
        "## 5.2. Los tres": (
            "Construiremos un único grafo con valores fijados de `M` y `efConstruction`, mediremos "
            "su construcción y barreremos `efSearch`. De ese modo, cualquier cambio online "
            "procederá de la amplitud de búsqueda y no de otro grafo diferente."
        ),
        "## 6.2. IVF-PQ": (
            "Entrenaremos IVF-PQ con la misma partición gruesa de 256 listas, 48 subcuantizadores "
            "y ocho bits por código. Después repetiremos el barrido de `nprobe`; si el recall se "
            "aplana, podremos atribuir el límite a la cuantización y no solo a las listas omitidas."
        ),
        "## 7.1. Recall": (
            "Reuniremos todos los registros en un DataFrame y dibujaremos recall frente a latencia, "
            "codificando además memoria y familia de índice. La misma tabla mostrará construcción y "
            "tamaño para que una curva rápida no oculte su coste offline."
        ),
        "## 7.2. Seleccionar": (
            "Fijaremos un recall@10 mínimo de 0,95 y filtraremos las configuraciones que lo cumplen. "
            "Entre ellas ordenaremos por latencia; el objetivo es convertir una preferencia vaga "
            "por la velocidad en una decisión condicionada y reproducible."
        ),
        "### Elegir entre post-filter": (
            "Simularemos un filtro de marca mediante oversampling sobre Flat. Pediremos más vecinos "
            "de los que mostraremos, hidrataremos sus metadatos y descartaremos los no permitidos "
            "para observar cuándo el post-filter consigue completar el top-k y cuándo no."
        ),
        "## 8.2. Persistencia": (
            "Cerraremos el ciclo escribiendo HNSW a disco, cargándolo en una instancia nueva y "
            "comparando sus primeros rankings con el original. La igualdad de IDs actuará como "
            "prueba mínima de integridad del artefacto serializado."
        ),
    }

    for cell in cells:
        if cell.cell_type != "markdown":
            continue
        bridge = next(
            (text for prefix, text in bridges.items() if cell.source.startswith(prefix)),
            None,
        )
        if bridge is not None:
            cell.source = f"{cell.source.rstrip()}\n\n{bridge}\n"
    return cells


def main() -> None:
    """Write the single notebook with stable metadata."""
    notebook = nbformat.v4.new_notebook(
        cells=add_execution_bridges(build_cells()),
        metadata={
            "kernelspec": {
                "display_name": "Python (BBDD Vectoriales · Sesión 2)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
            "case_study": "Marketplace español · FAISS exacto y ANN",
        },
    )
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"Created {NOTEBOOK_PATH.name} with {len(notebook.cells)} cells")


if __name__ == "__main__":
    main()
