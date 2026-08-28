"""Build the student-facing notebook for semantic product search."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT / "notebooks" / "sesion_01_buscador_semantico_ecommerce.ipynb"
)


def markdown(source: str) -> nbformat.NotebookNode:
    """Create a normalized Markdown cell."""
    return nbformat.v4.new_markdown_cell(dedent(source).strip() + "\n")


def code(source: str, *, tags: tuple[str, ...] = ()) -> nbformat.NotebookNode:
    """Create a normalized code cell with optional execution tags."""
    cell = nbformat.v4.new_code_cell(dedent(source).strip() + "\n")
    if tags:
        cell.metadata["tags"] = list(tags)
    return cell


def build_cells() -> list[nbformat.NotebookNode]:
    """Return the complete linear notebook."""
    cells: list[nbformat.NotebookNode] = []

    cells.extend(
        [
            markdown(
                r"""
                # Sesión práctica 1 · Búsqueda semántica en un marketplace

                ## De localizar palabras a comprender una necesidad

                Un buscador de productos recibe intenciones muy diferentes bajo la misma caja de texto. Algunas personas saben exactamente cómo se llama lo que quieren y escriben `taladro 24v batería`, `televisión 28 pulgadas` o el modelo concreto de un dispositivo. En esas consultas, las palabras, los números y las referencias son una señal excelente. Un sistema léxico bien construido puede resolverlas con enorme precisión.

                Otras personas no conocen el nombre comercial del producto. Describen una situación, una restricción o el resultado que persiguen: `necesito poner baldas sin hacer agujeros`, `quiero un ordenador que se convierta en tableta` o `busco legumbres aptas para una persona celíaca`. El catálogo puede contener exactamente los productos adecuados bajo los títulos *estante sin taladro*, *portátil convertible 2 en 1* y *lentejas sin gluten*. La consulta y el producto hablan de lo mismo, pero no emplean las mismas palabras.

                Ese desacoplamiento recibe el nombre de **vocabulary gap**. No aparece porque el buscador haya tokenizado mal. Aparece porque el lenguaje permite expresar una misma necesidad mediante sinónimos, perífrasis, consecuencias, unidades distintas y conocimiento implícito. Una persona dice *sin hacer agujeros* donde el catálogo dice *sin taladro*; dice *setenta centímetros* donde la ficha dice *28 pulgadas*; describe *apoyo para la espalda* donde el vendedor escribió *ergonómica*.

                El problema de negocio no consiste, por tanto, en reemplazar la búsqueda léxica por embeddings. Consiste en construir un ranking que conserve la precisión de marcas, modelos, medidas y negaciones, pero que también pueda recuperar por significado cuando el vocabulario cambia. Para llegar a esa decisión se estudiará cada capa del sistema: representación, geometría, función de similitud y evaluación.
                """
            ),
            markdown(
                r"""
                ## Índice de contenidos

                1. [El problema de negocio y los datos](#1-el-problema-de-negocio-y-los-datos)
                2. [La geometría que convierte vectores en rankings](#2-la-geometría-que-convierte-vectores-en-rankings)
                3. [Representaciones dispersas](#3-representaciones-dispersas)
                4. [Primeras representaciones densas](#4-primeras-representaciones-densas)
                5. [Representaciones contextuales](#5-representaciones-contextuales)
                6. [Modelos modernos de embeddings](#6-modelos-modernos-de-embeddings)
                7. [Más allá de un único vector denso](#7-más-allá-de-un-único-vector-denso)
                8. [Evaluación del buscador](#8-evaluación-del-buscador)
                9. [Selección de la arquitectura](#9-selección-de-la-arquitectura)
                """
            ),
            markdown(
                r"""
                ## 1. El problema de negocio y los datos

                ### 1.1. El contrato de relevancia

                El catálogo procede del **Shopping Queries Dataset** publicado por Amazon Science. La muestra local contiene 336 productos del locale español y 12 consultas con juicios de relevancia. Cada pareja consulta-producto tiene una etiqueta ESCI:

                - `E · Exact`: el producto satisface de forma precisa la intención.
                - `S · Substitute`: no es el resultado exacto, pero podría sustituirlo razonablemente.
                - `C · Complement`: acompaña al producto buscado, pero no lo sustituye.
                - `I · Irrelevant`: no resuelve la necesidad.

                La distinción entre sustituto y complemento es especialmente valiosa. Para `funda iPad Air 4 sin tapa`, una funda compatible con tapa puede ser un sustituto imperfecto; un protector de pantalla es un complemento; un cargador de portátil es irrelevante. Un sistema binario que redujera todo a relevante/no relevante perdería esos matices.

                Además de las consultas originales se utilizarán ocho **paráfrasis de estrés**. Cada paráfrasis conserva deliberadamente la intención de una consulta original, pero cambia su superficie léxica. Por eso reutiliza el mismo conjunto de candidatos y los mismos juicios ESCI. Este conjunto no pretende sustituir una evaluación de producción: aísla una pregunta concreta y medible, cuánto cae cada representación cuando desaparecen las palabras obvias.
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
                from vector_search_session.ecommerce import (
                    ESCI_GAINS,
                    ESCI_LABEL_NAMES,
                    load_esci_sample,
                )

                sample = load_esci_sample()
                products = sample.products.copy()
                judgments = sample.judgments.copy()
                semantic_queries = pd.read_csv(
                    project_root / "data/esci/semantic_queries.csv"
                )
                """
            ),
            code(
                """
                unique_queries = (
                    judgments[["query_id", "query"]]
                    .drop_duplicates()
                    .sort_values("query_id", ignore_index=True)
                )

                print(f"Productos: {len(products):,}")
                print(f"Consultas originales: {len(unique_queries):,}")
                print(f"Paráfrasis de estrés: {len(semantic_queries):,}")
                print(f"Juicios consulta-producto: {len(judgments):,}")
                """
            ),
            code(
                """
                semantic_queries[
                    ["original_query", "semantic_query", "lexical_gap"]
                ].style.set_properties(subset=["semantic_query"], **{"font-weight": "bold"})
                """
            ),
            markdown(
                r"""
                La tabla anterior hace visible el problema. `Sillas oficina ergonómicas` no es, por sí sola, una demostración convincente de búsqueda semántica: comparte los términos principales con muchas fichas relevantes y un baseline léxico debería funcionar bien. Su pareja `necesito un asiento cómodo para trabajar ocho horas con buen apoyo para la espalda` es distinta. Mantiene la necesidad, pero sustituye la categoría por una situación de uso y los atributos por sus consecuencias.

                Esta separación evita una conclusión tramposa. Si un embedding mejora la consulta literal, puede deberse a muchas razones. Si mantiene la calidad cuando se introduce una paráfrasis y TF-IDF cae, se ha observado de forma mucho más limpia la capacidad que se quería estudiar. Aun así, el sistema semántico no recibe un cheque en blanco: también debe sobrevivir a números, negaciones, marcas y productos complementarios.

                La evaluación final mantendrá ambas vistas. Las consultas originales representan el comportamiento real publicado en ESCI. Las paráfrasis representan un *slice* adversarial construido para medir robustez semántica. Un modelo solo será una opción razonable si se entiende qué gana en una vista y qué pierde en la otra.
                """
            ),
            code(
                """
                label_summary = (
                    judgments["esci_label"]
                    .value_counts()
                    .rename_axis("label")
                    .reset_index(name="pairs")
                )
                label_summary["meaning"] = label_summary["label"].map(
                    ESCI_LABEL_NAMES
                )
                label_summary
                """
            ),
            code(
                """
                label_figure = px.bar(
                    label_summary,
                    x="label",
                    y="pairs",
                    color="meaning",
                    text="pairs",
                    color_discrete_sequence=px.colors.qualitative.Safe,
                )
                label_figure.update_layout(
                    title="Distribución de relevancia graduada en la muestra",
                    xaxis_title="Etiqueta ESCI",
                    yaxis_title="Pares consulta-producto",
                )
                label_figure.show()
                """
            ),
            markdown(
                r"""
                ### 1.2. Qué representa exactamente a un producto

                Un modelo de embeddings no recibe un objeto `producto`. Recibe una secuencia. La ficha original separa título, marca, color, bullets y descripción; algunas columnas están vacías y otras repiten información. La función `compose_product_text` convierte esos campos en una representación textual reproducible.

                Esta operación no es una limpieza administrativa previa al modelado. Es parte del modelo de recuperación. Repetir tres veces la marca aumenta su presencia en una representación léxica. Anteponer `Marca:` o `Color:` aporta estructura que algunos encoders pueden aprovechar. Incluir una descripción larguísima puede diluir el título, superar el contexto máximo o truncar precisamente la información que importaba.

                La versión utilizada concatena título, marca, color, bullets y descripción en ese orden. Si cambia el orden, la plantilla o el tratamiento de nulos, cambia el corpus que se embebe. Por eso un índice reproducible necesita guardar junto a los vectores el identificador del modelo, su dimensión, la normalización y la plantilla exacta de entrada.
                """
            ),
            code(
                """
                product_example = products.loc[
                    products["product_title"].str.contains("ergon", case=False, na=False)
                ].iloc[0]

                product_example[
                    [
                        "product_title",
                        "product_brand",
                        "product_color",
                        "product_bullet_point",
                        "product_description",
                    ]
                ]
                """
            ),
            code(
                """
                print(product_example["searchable_text"][:1_200])
                """
            ),
        ]
    )

    cells.extend(
        [
            markdown(
                r"""
                # 2. La geometría que convierte vectores en rankings

                Un embedding es una función $f$ que transforma una entrada en un vector real:

                $$
                f(x)=\mathbf{x}\in\mathbb{R}^{d}
                $$

                El vector no contiene una etiqueta legible en cada coordenada. En modelos aprendidos, una dimensión aislada rara vez significa *ergonomía* o *color rojo*. La información se distribuye por el espacio y adquiere sentido mediante relaciones entre vectores. Para recuperar productos se necesita otra función $s(\mathbf{q},\mathbf{x})$ que compare la consulta con cada candidato y produzca un orden.

                Conviene separar ambas decisiones. El encoder determina la geometría que intenta aprender; la métrica decide cómo se recorre esa geometría. Cambiar de coseno a producto escalar sin comprender el entrenamiento puede alterar el ranking tanto como cambiar de modelo. Del mismo modo, normalizar los vectores no es una optimización inocente: decide que la magnitud dejará de aportar información.

                El siguiente espacio bidimensional es deliberadamente interpretable. Sus ejes representan afinidad con *asiento* y con *trabajo/ergonomía*. Un modelo real usará cientos o miles de dimensiones sin nombres, pero el producto escalar, la norma y la distancia se calculan exactamente igual.
                """
            ),
            code(
                """
                vector_names = [
                    "silla ergonómica",
                    "silla gaming",
                    "silla de comedor",
                    "cojín lumbar",
                    "mesa de oficina",
                ]
                product_vectors = np.array(
                    [[0.95, 0.92], [0.90, 0.62], [0.82, 0.15], [0.38, 0.88], [0.12, 0.74]],
                    dtype=np.float32,
                )
                query_vector = np.array([0.92, 0.90], dtype=np.float32)
                """
            ),
            code(
                """
                vector_figure = go.Figure()
                for product_name, product_vector in zip(
                    vector_names, product_vectors, strict=True
                ):
                    vector_figure.add_trace(
                        go.Scatter(
                            x=[0, product_vector[0]],
                            y=[0, product_vector[1]],
                            name=product_name,
                        )
                    )
                """
            ),
            code(
                """
                vector_figure.add_trace(
                    go.Scatter(
                        x=[0, query_vector[0]],
                        y=[0, query_vector[1]],
                        name="consulta",
                        line={"width": 6, "color": "#ef4444"},
                    )
                )
                vector_figure.update_layout(
                    title="Consulta y candidatos en un espacio de dos dimensiones",
                    xaxis_title="Afinidad con 'asiento'",
                    yaxis_title="Afinidad con 'trabajo / ergonomía'",
                    height=520,
                )
                vector_figure.show()
                """
            ),
            markdown(
                r"""
                ## 2.1. Coseno, producto escalar y distancia euclídea

                El **producto escalar** multiplica coordenadas correspondientes y las suma:

                $$\mathbf{q}^{\top}\mathbf{x}=\sum_{i=1}^{d}q_i x_i$$

                Su valor aumenta cuando los vectores están alineados, pero también cuando crecen sus normas. Puede interpretarse como dirección multiplicada por magnitud, porque

                $$\mathbf{q}^{\top}\mathbf{x}=\lVert\mathbf{q}\rVert_2\lVert\mathbf{x}\rVert_2\cos(\theta).$$

                La **similitud coseno** divide por ambas normas y conserva solo el ángulo. Dos vectores paralelos obtienen coseno 1 aunque uno sea cien veces más largo. Esto resulta útil cuando la magnitud depende de factores espurios, pero elimina cualquier información que el modelo hubiera codificado en ella.

                La **distancia euclídea** mide la longitud del desplazamiento entre los puntos:

                $$d_{L2}(\mathbf{q},\mathbf{x})=\sqrt{\sum_{i=1}^{d}(q_i-x_i)^2}.$$

                En similitudes, un valor mayor suele ser mejor; en distancias, un valor menor es mejor. Una implementación puede convertir la distancia en score mediante su negativo para mantener una interfaz de orden descendente, pero ese cambio de signo no modifica la geometría.
                """
            ),
            code(
                """
                from vector_search_session.distances import compute_vector_scores

                metric_rows = []
                for metric_name in ["cosine", "dot", "l2"]:
                    metric_scores = compute_vector_scores(
                        product_vectors, query_vector, metric=metric_name
                    )
                    for product_name, score in zip(
                        vector_names, metric_scores, strict=True
                    ):
                        metric_rows.append(
                            {"metric": metric_name, "product": product_name, "score": score}
                        )
                """
            ),
            code(
                """
                metric_frame = pd.DataFrame(metric_rows)
                metric_frame.pivot(index="product", columns="metric", values="score")
                """
            ),
            markdown(
                r"""
                ## 2.2. La norma puede cambiar por completo el resultado

                Supóngase un candidato que apunta exactamente en la dirección de la consulta, pero cuya norma es cinco veces mayor. Para coseno representa la misma dirección semántica. Para producto escalar recibe una puntuación cinco veces mayor. Para L2 se encuentra lejos, porque la distancia absoluta entre ambos puntos ha crecido.

                Este comportamiento no es un defecto matemático. Cada métrica responde a una pregunta distinta. El defecto aparece cuando se elige una sin respetar el contrato del modelo. Algunos sistemas de recomendación permiten que la norma capture popularidad o confianza y entrenan explícitamente con producto escalar. Muchos encoders de texto, en cambio, se evalúan con coseno o entregan vectores normalizados.

                También existe un efecto operativo. Si las normas varían mucho por longitud del documento o por frecuencia, el producto escalar puede favorecer sistemáticamente ciertos productos aunque su dirección no sea la mejor. Inspeccionar la distribución de normas es, por tanto, una prueba de calidad de datos, no una curiosidad geométrica.
                """
            ),
            code(
                """
                scaled_query = query_vector * 5
                extended_vectors = np.vstack([product_vectors, scaled_query])
                extended_names = [*vector_names, "consulta escalada x5"]
                """
            ),
            code(
                """
                norm_trap_rows = []
                for metric_name in ["cosine", "dot", "l2"]:
                    metric_scores = compute_vector_scores(
                        extended_vectors, query_vector, metric=metric_name
                    )
                    for product_name, score in zip(
                        extended_names, metric_scores, strict=True
                    ):
                        norm_trap_rows.append(
                            {"metric": metric_name, "product": product_name, "score": score}
                        )
                """
            ),
            code(
                """
                pd.DataFrame(norm_trap_rows).pivot(
                    index="product", columns="metric", values="score"
                )
                """
            ),
            markdown(
                r"""
                ## 2.3. Normalización L2 y equivalencia de rankings

                Normalizar un vector consiste en dividirlo por su norma:

                $$\widehat{\mathbf{x}}=\frac{\mathbf{x}}{\lVert\mathbf{x}\rVert_2}.$$

                El resultado vive sobre la superficie de una hiperesfera de radio uno. Si consulta y documentos están normalizados, el denominador del coseno vale uno y la similitud coseno coincide exactamente con el producto escalar. Además,

                $$\lVert\widehat{\mathbf{q}}-\widehat{\mathbf{x}}\rVert_2^2=2-2\widehat{\mathbf{q}}^{\top}\widehat{\mathbf{x}}.$$

                Por tanto, L2, coseno y dot producen el mismo ranking sobre vectores unitarios, aunque sus valores y el sentido de ordenación sean distintos. Esta equivalencia permite usar una multiplicación matricial para calcular coseno y evita normalizar cada candidato en cada consulta.

                La normalización debe realizarse tanto al indexar como al consultar. Normalizar solo una de las dos partes rompe la equivalencia. Normalizar un vector nulo produce una división por cero, de modo que una implementación robusta debe detectar entradas vacías o respuestas inválidas antes de construir el índice.
                """
            ),
            code(
                """
                from vector_search_session.distances import safe_l2_normalize

                normalized_products = safe_l2_normalize(extended_vectors, axis=1)
                normalized_query = safe_l2_normalize(query_vector)
                normalized_cosine = compute_vector_scores(
                    normalized_products, normalized_query, metric="cosine"
                )
                normalized_dot = compute_vector_scores(
                    normalized_products, normalized_query, metric="dot"
                )
                """
            ),
            code(
                """
                np.testing.assert_allclose(normalized_cosine, normalized_dot, atol=1e-6)
                print("Coseno y producto escalar coinciden sobre vectores unitarios.")
                """
            ),
            markdown(
                r"""
                ## 2.4. Dimensionalidad: capacidad, coste y concentración

                La dimensión $d$ determina cuántos números almacena cada vector. Aumentarla da al modelo más capacidad para distribuir información, pero no garantiza que esa capacidad se utilice bien. Un millón de vectores `float32` de 384 dimensiones ocupan aproximadamente 1,43 GiB antes de añadir el índice; con 3.072 dimensiones, unos 11,44 GiB. El coste del producto escalar también crece linealmente con $d$.

                En puntos aleatorios de alta dimensión aparece la **concentración de distancias**: el vecino más próximo y el más lejano se parecen cada vez más en términos relativos. El experimento siguiente mide la razón entre distancia mínima y máxima. Cuando se acerca a uno, las distancias discriminan menos.

                Los embeddings aprendidos no son gaussianas aleatorias y este gráfico no demuestra que una dimensión alta empeore la búsqueda. Explica por qué `más dimensiones` no puede utilizarse como sinónimo de `mejor modelo`. La dimensión correcta se decide midiendo calidad, memoria y latencia en el mismo problema.
                """
            ),
            code(
                """
                random_generator = np.random.default_rng(42)
                dimension_rows = []
                for dimension in [2, 8, 32, 128, 512, 2_048]:
                    random_products = random_generator.normal(size=(2_000, dimension))
                    random_query = random_generator.normal(size=dimension)
                    distances = np.linalg.norm(random_products - random_query, axis=1)
                    dimension_rows.append(
                        {
                            "dimension": dimension,
                            "nearest_over_farthest": distances.min() / distances.max(),
                        }
                    )
                """
            ),
            code(
                """
                dimension_frame = pd.DataFrame(dimension_rows)
                dimension_figure = px.line(
                    dimension_frame,
                    x="dimension",
                    y="nearest_over_farthest",
                    markers=True,
                    log_x=True,
                )
                dimension_figure.update_layout(
                    title="Concentración de distancias en puntos aleatorios",
                    yaxis_title="Distancia mínima / distancia máxima",
                )
                dimension_figure.show()
                """
            ),
        ]
    )

    cells.extend(
        [
            markdown(
                r"""
                # 3. Representaciones dispersas

                ## 3.1. Bag-of-Words: una coordenada por término

                **Bag-of-Words** construye un vocabulario con los términos del corpus y asigna una dimensión a cada uno. Si el vocabulario contiene `silla`, `ergonómica`, `mesa` y `roja`, el texto `silla roja roja` puede representarse como $[1,0,0,2]$. La posición de las palabras desaparece: `silla no ergonómica` y `no silla ergonómica` contienen los mismos unigramas.

                La dimensionalidad de estos vectores puede ser enorme, pero cada documento activa una fracción diminuta del vocabulario. Un producto con 40 términos distintos dentro de un vocabulario de 50.000 tiene aproximadamente un 99,92 % de ceros. Guardar una matriz densa desperdiciaría memoria; formatos como CSR almacenan los valores no nulos, sus columnas y los punteros que delimitan cada fila.

                Esta representación es **dispersa** por su patrón de ceros, no porque tenga pocas dimensiones. Un embedding de 384 dimensiones suele ser denso porque casi todas sus coordenadas son distintas de cero. La distinción afecta al almacenamiento, a las operaciones eficientes y al tipo de índice que puede utilizarse.
                """
            ),
            code(
                """
                from sklearn.feature_extraction.text import CountVectorizer

                count_vectorizer = CountVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 1),
                    min_df=2,
                )
                count_matrix = count_vectorizer.fit_transform(
                    products["searchable_text"]
                )
                """
            ),
            code(
                """
                total_positions = count_matrix.shape[0] * count_matrix.shape[1]
                density = count_matrix.nnz / total_positions
                csr_bytes = (
                    count_matrix.data.nbytes
                    + count_matrix.indices.nbytes
                    + count_matrix.indptr.nbytes
                )
                dense_bytes = total_positions * np.dtype(np.int64).itemsize

                print(f"Matriz: {count_matrix.shape}")
                print(f"Densidad: {density:.2%}")
                print(f"CSR: {csr_bytes / 2**20:.2f} MiB")
                print(f"Densa equivalente: {dense_bytes / 2**20:.2f} MiB")
                """
            ),
            markdown(
                r"""
                ## 3.2. TF-IDF: distinguir una coincidencia informativa de una trivial

                El conteo bruto favorece palabras repetidas y no distingue entre términos comunes y discriminativos. TF-IDF combina dos factores. La frecuencia de término $tf(t,d)$ mide cuánto aparece $t$ en el documento $d$. La frecuencia inversa de documento reduce el peso de términos presentes en gran parte de la colección:

                $$idf(t)=\log\left(\frac{N+1}{df(t)+1}\right)+1.$$

                `iPad Air 4` puede ser muy informativo porque aparece en pocas fichas. `Producto`, `calidad` o `para` apenas ayudan a separar candidatos. Con `sublinear_tf=True`, la frecuencia se transforma aproximadamente en $1+\log(tf)$ para impedir que repetir un término veinte veces multiplique linealmente su influencia.

                `TfidfVectorizer` normaliza cada fila con L2. Gracias a ello, multiplicar la matriz de documentos por el vector columna de la consulta calcula directamente la similitud coseno sparse.
                """
            ),
            code(
                """
                from sklearn.feature_extraction.text import TfidfVectorizer

                tfidf_vectorizer = TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 1),
                    sublinear_tf=True,
                    norm="l2",
                )
                tfidf_matrix = tfidf_vectorizer.fit_transform(
                    products["searchable_text"]
                )
                print(f"Shape TF-IDF: {tfidf_matrix.shape}")
                """
            ),
            markdown(
                r"""
                ## 3.3. La consulta debe atravesar exactamente la misma transformación

                `fit_transform` aprende el vocabulario y los valores IDF exclusivamente a partir del catálogo. Una consulta nueva utiliza `transform`: no amplía el vocabulario ni recalcula los pesos. Si contiene una palabra desconocida, esa palabra no activa ninguna coordenada. Esta asimetría es correcta; cambiar el espacio con cada consulta invalidaría todos los documentos indexados.

                La consulta literal `television 28 pulgadas` comparte una categoría y una medida exacta con los productos. TF-IDF debería comportarse bien. La paráfrasis `busco un televisor pequeño de unas setenta centímetros para la cocina` cambia la forma de la categoría, convierte la medida a otra unidad y expresa el tamaño como una cualidad. El sistema léxico no sabe que 28 pulgadas son aproximadamente 71 centímetros ni que ese formato suele corresponder a un televisor pequeño.

                Esta limitación no convierte TF-IDF en un mal baseline. Su incapacidad semántica es también una fuente de precisión: no inventa que `24V` y `18V` son equivalentes, ni que una funda `con tapa` satisface una consulta `sin tapa`. El objetivo posterior será complementar esa señal, no borrarla.
                """
            ),
            code(
                """
                demonstration_query_id = 101_352
                original_text = semantic_queries.loc[
                    semantic_queries["query_id"] == demonstration_query_id,
                    "original_query",
                ].iloc[0]
                semantic_text = semantic_queries.loc[
                    semantic_queries["query_id"] == demonstration_query_id,
                    "semantic_query",
                ].iloc[0]
                """
            ),
            code(
                """
                original_tfidf = tfidf_vectorizer.transform([original_text])
                semantic_tfidf = tfidf_vectorizer.transform([semantic_text])
                original_scores = (tfidf_matrix @ original_tfidf.T).toarray().ravel()
                semantic_scores = (tfidf_matrix @ semantic_tfidf.T).toarray().ravel()
                """
            ),
            code(
                """
                def top_products(
                    scores: np.ndarray,
                    limit: int = 6,
                    query_id: int | None = None,
                ) -> pd.DataFrame:
                    top_indices = np.argsort(-scores, kind="stable")[:limit]
                    result = products.iloc[top_indices][
                        ["product_id", "product_title", "product_brand"]
                    ].copy()
                    result["score"] = scores[top_indices]
                    if query_id is not None:
                        query_labels = judgments.loc[
                            judgments["query_id"] == query_id,
                            ["product_id", "esci_label"],
                        ]
                        result = result.merge(query_labels, on="product_id", how="left")
                    return result
                """
            ),
            code(
                """
                pd.concat(
                    {
                        "consulta literal": top_products(
                            original_scores, query_id=demonstration_query_id
                        ),
                        "paráfrasis": top_products(
                            semantic_scores, query_id=demonstration_query_id
                        ),
                    },
                    names=["entrada", "fila"],
                )
                """
            ),
            markdown(
                r"""
                El cambio de ranking debe interpretarse con los títulos, no solo con los scores. En TF-IDF, cero significa ausencia de características compartidas; un valor alto significa mucho solapamiento ponderado. No significa probabilidad de compra ni porcentaje de relevancia.

                Los modelos densos intentarán colocar cerca expresiones como *televisión de 28 pulgadas*, *televisor pequeño* y *pantalla de unos setenta centímetros*. No ejecutan una conversión de unidades fiable como una calculadora; aprenden asociaciones distribucionales que pueden cerrar parte de esa distancia. Para comprender cómo lo hacen conviene recorrer la evolución desde vectores de palabra estáticos hasta encoders contrastivos de frases.
                """
            ),
        ]
    )

    cells.extend(
        [
            markdown(
                r"""
                # 4. Primeras representaciones densas

                ## 4.1. Word2Vec: aprender una palabra por los contextos que la rodean

                Word2Vec no parte de un diccionario de sinónimos. Construye pares de entrenamiento recorriendo ventanas del corpus. En **Skip-gram**, dada una palabra central $w_t$, intenta predecir las palabras de contexto $w_{t+j}$. En **CBOW** hace lo contrario: combina el contexto para predecir la palabra central.

                La formulación completa con softmax requiere normalizar sobre todo el vocabulario en cada ejemplo. **Negative sampling** sustituye ese coste por una tarea binaria. Para un par real $(w,c)$ se maximiza $\log\sigma(\mathbf{v}_c^\top\mathbf{v}_w)$; para varios contextos negativos $n$ se maximiza $\log\sigma(-\mathbf{v}_n^\top\mathbf{v}_w)$. Las palabras que comparten contextos reciben actualizaciones parecidas y terminan cerca.

                El resultado es una tabla con un vector por entrada del vocabulario. El vector de `banco` es siempre el mismo, tanto en una entidad financiera como en un asiento. Word2Vec aprende regularidades distribucionales, pero no contextualiza cada aparición. Tampoco produce directamente un vector de producto: promediar las palabras es una heurística que pierde orden y asigna el mismo peso a todas salvo que se introduzca otra ponderación.
                """
            ),
            code(
                """
                from gensim.models import FastText, Word2Vec
                from gensim.utils import simple_preprocess

                tokenized_products = [
                    simple_preprocess(product_text, deacc=True)
                    for product_text in products["searchable_text"]
                ]
                """
            ),
            code(
                """
                word2vec_model = Word2Vec(
                    sentences=tokenized_products,
                    vector_size=64,
                    window=5,
                    min_count=2,
                    sg=1,
                    negative=8,
                    workers=1,
                    seed=42,
                    epochs=40,
                )
                """
            ),
            code(
                """
                word2vec_model.wv.most_similar("silla", topn=8)
                """
            ),
            markdown(
                r"""
                El entrenamiento anterior utiliza solo 336 productos. Sus vecinos no deben interpretarse como la calidad alcanzable por Word2Vec con miles de millones de tokens; sirven para observar el mecanismo y también su dependencia del corpus. Si el catálogo contiene `silla` sobre todo junto a `gaming`, el espacio reflejará esa distribución, no una definición universal de silla.

                ## 4.2. GloVe: convertir coapariciones globales en geometría

                GloVe parte de una matriz $X_{ij}$ que cuenta cuántas veces aparece la palabra $j$ en el contexto de $i$. Aprende vectores minimizando una regresión ponderada:

                $$J=\sum_{i,j}f(X_{ij})\left(\mathbf{w}_i^\top\widetilde{\mathbf{w}}_j+b_i+\widetilde{b}_j-\log X_{ij}\right)^2.$$

                El logaritmo comprime conteos extremos y $f(X_{ij})$ reduce el efecto de pares muy raros o excesivamente frecuentes. La intuición central es que las **proporciones** de coaparición codifican relaciones. Una palabra que aparece cerca de `oficina`, `lumbar` y `ajustable` mantiene un patrón distinto de otra asociada a `comedor`, aunque ambas sean sillas.

                Word2Vec optimiza una tarea predictiva sobre muestras locales; GloVe factoriza información global agregada. Sus detalles difieren, pero comparten dos limitaciones para este buscador: un vector estático por palabra y la necesidad de fabricar después una representación para la secuencia completa.
                """
            ),
            code(
                """
                classic_models = pd.DataFrame(
                    [
                        ["Word2Vec", "predicción de contexto", "palabra", "no"],
                        ["GloVe", "coaparición global", "palabra", "no"],
                        ["FastText", "contexto + subpalabras", "palabra", "parcial"],
                    ],
                    columns=["modelo", "señal", "unidad", "fuera_de_vocabulario"],
                )
                classic_models
                """
            ),
            markdown(
                r"""
                ## 4.3. FastText: representar una palabra mediante subpalabras

                FastText conserva el objetivo predictivo de Word2Vec, pero representa cada palabra como la suma de un vector propio y vectores de n-gramas de caracteres. Con marcadores de inicio y final, `silla` puede descomponerse en fragmentos como `<si`, `sil`, `ill`, `lla` y `la>`. `sillas` comparte gran parte de esas piezas; una errata como `ergonomikas` puede recibir un vector aunque no apareciera exactamente en el entrenamiento.

                Esta capacidad es importante en catálogos: variantes morfológicas, errores, referencias y palabras compuestas son frecuentes. Sin embargo, generar un vector para una cadena desconocida no implica comprenderla. FastText extrapola desde su forma. Si dos palabras se parecen ortográficamente pero significan cosas distintas, la señal de subpalabras también puede inducir errores.

                El ejemplo compara una palabra ausente en Word2Vec con el vector sintetizado por FastText. El resultado debe leerse como robustez de vocabulario, no como prueba de comprensión contextual.
                """
            ),
            code(
                """
                fasttext_model = FastText(
                    sentences=tokenized_products,
                    vector_size=64,
                    window=5,
                    min_count=2,
                    sg=1,
                    negative=8,
                    workers=1,
                    seed=42,
                    epochs=40,
                )
                """
            ),
            code(
                """
                misspelled_word = "ergonomikas"
                print("Word2Vec conoce la cadena:", misspelled_word in word2vec_model.wv)
                print("FastText puede representarla:", misspelled_word in fasttext_model.wv)
                fasttext_model.wv.most_similar(misspelled_word, topn=5)
                """
            ),
            markdown(
                r"""
                # 5. Representaciones contextuales

                Word2Vec, GloVe y FastText han permitido aprender relaciones que los conteos no contenían, pero los tres mantienen una representación fija para cada palabra. El siguiente paso consiste en dejar que el contexto modifique el vector de cada aparición.

                ## 5.1. BERT: el vector de un token depende de toda la secuencia

                BERT sustituye la tabla estática por un Transformer bidireccional. Cada token se convierte primero en la suma de embedding de token, posición y segmento. Después atraviesa capas de **self-attention**. Para cada posición se calculan consulta, clave y valor:

                $$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.$$

                La matriz $QK^\top$ permite que cada token pondere a los demás. En `banco de madera para el jardín`, la representación final de `banco` incorpora `madera` y `jardín`; en `banco que concede hipotecas`, incorpora un contexto financiero. Las cabezas múltiples pueden especializarse en relaciones diferentes y las capas sucesivas mezclan información cada vez más contextual.

                El BERT original se preentrena enmascarando tokens y prediciéndolos desde ambos lados. Esa tarea produce excelentes representaciones para adaptar a clasificación, extracción o preguntas y respuestas. No obliga, sin embargo, a que el coseno entre dos vectores de frase sea una buena medida de relevancia. Usar el token `[CLS]` o promediar tokens de un BERT genérico crea un vector, pero no garantiza una geometría adecuada para recuperación.
                """
            ),
            markdown(
                r"""
                ## 5.2. Sentence-BERT: entrenar explícitamente el espacio de frases

                Sentence-BERT utiliza dos ramas con pesos compartidos. Una codifica la consulta y otra el documento; sobre las representaciones de tokens se aplica una estrategia de *pooling*, normalmente media, para obtener un solo vector por secuencia. El entrenamiento utiliza pares o tripletas y una pérdida que acerca ejemplos positivos y separa negativos.

                En una pérdida contrastiva con negativos dentro del batch, para la consulta $q_i$ y su documento positivo $d_i$ puede minimizarse:

                $$
                \mathcal{L}_i=-\log\frac{\exp(s(q_i,d_i)/\tau)}
                {\sum_j\exp(s(q_i,d_j)/\tau)}.
                $$

                La temperatura $\tau$ controla cuánto se acentúan las diferencias de similitud. Los documentos positivos deben competir contra otros documentos del batch. Los **hard negatives** —productos plausibles pero incorrectos— enseñan fronteras más finas que negativos aleatorios obvios.

                La arquitectura recibe el nombre de **bi-encoder** porque consulta y documento se codifican por separado. Esto permite calcular el catálogo una vez y recuperar mediante productos escalares. Un **cross-encoder** concatena consulta y documento, deja que todos sus tokens interactúen en cada capa y suele ser más preciso, pero debe ejecutarse para cada pareja. Por eso suele utilizarse para reordenar decenas de candidatos, no millones.
                """
            ),
            code(
                """
                catalog_sizes = np.array([1_000, 10_000, 100_000, 1_000_000])
                architecture_frame = pd.DataFrame(
                    {
                        "productos": np.tile(catalog_sizes, 2),
                        "codificaciones_online": np.concatenate(
                            [np.ones(4), catalog_sizes]
                        ),
                        "arquitectura": ["Bi-encoder"] * 4 + ["Cross-encoder"] * 4,
                    }
                )
                """
            ),
            code(
                """
                architecture_figure = px.line(
                    architecture_frame,
                    x="productos",
                    y="codificaciones_online",
                    color="arquitectura",
                    markers=True,
                    log_x=True,
                    log_y=True,
                )
                architecture_figure.update_layout(
                    title="Trabajo online conceptual por consulta",
                    yaxis_title="Pares que deben codificarse online",
                )
                architecture_figure.show()
                """
            ),
        ]
    )

    cells.extend(
        [
            markdown(
                r"""
                # 6. Modelos modernos de embeddings

                Sentence-BERT establece la arquitectura que hace viable comparar secuencias completas, pero no designa un único modelo. A partir de esa idea aparecen familias entrenadas con más datos, idiomas y objetivos de recuperación. Empezaremos por una alternativa local y continuaremos con servicios gestionados sometidos al mismo protocolo.

                ## 6.1. Un encoder open-weight: multilingual E5

                E5 formula numerosas tareas de NLP como pares `texto 1`, `texto 2` y se entrena con aprendizaje contrastivo. En recuperación asimétrica, la consulta suele ser corta y el documento más largo; no desempeñan el mismo papel. Por eso la familia E5 antepone `query:` a las consultas y `passage:` a los documentos. Los prefijos forman parte de la distribución de entrenamiento, no son comentarios para humanos.

                Se utilizará `intfloat/multilingual-e5-small`, de 384 dimensiones. Los vectores se han calculado previamente con normalización L2 para que el notebook sea reproducible y la primera comparación no dependa de una descarga. El script de generación permanece disponible y contiene exactamente la plantilla aplicada.

                Al cargar embeddings precalculados deben validarse dos contratos. Primero, el orden de los `product_id` debe coincidir con el DataFrame; una matriz correcta alineada con IDs incorrectos genera resultados silenciosamente absurdos. Segundo, la forma y la distribución de normas deben coincidir con los metadatos del índice.
                """
            ),
            code(
                """
                product_archive = np.load(project_root / "data/esci/e5_products.npz")
                query_archive = np.load(project_root / "data/esci/e5_queries.npz")
                semantic_archive = np.load(
                    project_root / "data/esci/e5_semantic_queries.npz"
                )

                e5_product_ids = product_archive["product_ids"].astype(str)
                e5_product_embeddings = product_archive["embeddings"]
                e5_query_ids = query_archive["query_ids"]
                e5_query_embeddings = query_archive["embeddings"]
                e5_semantic_ids = semantic_archive["query_ids"]
                e5_semantic_embeddings = semantic_archive["embeddings"]
                """
            ),
            code(
                """
                expected_product_ids = products["product_id"].astype(str).to_numpy()
                np.testing.assert_array_equal(e5_product_ids, expected_product_ids)

                product_norms = np.linalg.norm(e5_product_embeddings, axis=1)
                print("Shape:", e5_product_embeddings.shape)
                print("Norma mínima:", product_norms.min())
                print("Norma máxima:", product_norms.max())
                """
            ),
            markdown(
                r"""
                ### La prueba decisiva: misma intención, vocabulario diferente

                Se recuperarán productos para la pareja de televisores mediante TF-IDF y E5. La consulta literal favorece al baseline léxico porque contiene `television`, `28` y `pulgadas`. La paráfrasis cambia la unidad, introduce el contexto de uso y evita la expresión exacta de la ficha.

                E5 no consulta un tesauro ni una tabla de conversión durante la búsqueda. El encoder transforma la secuencia completa y coloca cerca expresiones que durante el entrenamiento funcionaron como pares relacionados o aparecieron en contextos semejantes. La relación está distribuida en el vector; no existe una coordenada `28 pulgadas equivalen a 71 centímetros` que pueda inspeccionarse directamente.

                El score denso tampoco es una probabilidad. Un coseno de 0,86 no significa 86 % de relevancia. Sirve para ordenar elementos dentro del mismo espacio y configuración. Comparar el 0,86 de E5 con el 0,72 de otro modelo carece de sentido si los espacios y distribuciones de scores son diferentes.
                """
            ),
            code(
                """
                semantic_position = np.flatnonzero(
                    e5_semantic_ids == demonstration_query_id
                ).item()
                e5_semantic_query = e5_semantic_embeddings[semantic_position]
                e5_semantic_scores = e5_product_embeddings @ e5_semantic_query
                """
            ),
            code(
                """
                pd.concat(
                    {
                        "TF-IDF · paráfrasis": top_products(
                            semantic_scores, query_id=demonstration_query_id
                        ),
                        "E5 · paráfrasis": top_products(
                            e5_semantic_scores, query_id=demonstration_query_id
                        ),
                    },
                    names=["sistema", "fila"],
                )
                """
            ),
            markdown(
                r"""
                ## 6.3. Matryoshka y reducción de dimensionalidad

                Algunos encoders modernos se entrenan con **Matryoshka Representation Learning**. La pérdida se aplica a varios prefijos del mismo vector para que las primeras 256, 512 o 768 dimensiones formen representaciones útiles anidadas dentro del vector completo. Esto permite elegir un punto de coste-calidad sin entrenar un modelo diferente para cada dimensión.

                No debe truncarse cualquier embedding arbitrariamente. La propiedad depende del entrenamiento y de las dimensiones soportadas por el proveedor. Tras truncar, los modelos antiguos pueden requerir normalización L2 manual. Además, cambiar la dimensión modifica la forma del índice: consultas y documentos deben regenerarse con exactamente la misma configuración.

                El ahorro se calcula antes de considerar la sobrecarga del índice. Con `float32`, cada coordenada ocupa cuatro bytes. La cuantización puede reducir ese tamaño a `float16`, `int8` o incluso representaciones binarias, pero introduce otro compromiso de calidad que también debe evaluarse.
                """
            ),
            code(
                """
                dimension_options = np.array([256, 384, 512, 768, 1_024, 1_536, 3_072])
                storage_frame = pd.DataFrame(
                    {
                        "dimension": dimension_options,
                        "GiB_por_millon": dimension_options * 4 * 1_000_000 / 2**30,
                    }
                )
                """
            ),
            code(
                """
                storage_figure = px.bar(
                    storage_frame,
                    x="dimension",
                    y="GiB_por_millon",
                    text_auto=".1f",
                )
                storage_figure.update_layout(
                    title="Almacenamiento bruto de un millón de vectores float32",
                    yaxis_title="GiB sin contar el índice",
                )
                storage_figure.show()
                """
            ),
        ]
    )

    cells.extend(
        [
            markdown(
                r"""
                ## 6.2. Modelos de embeddings mediante API

                Los proveedores gestionados eliminan la operación del modelo y ofrecen endpoints versionados, escalado y facturación por uso. A cambio, introducen dependencia externa, coste variable, límites de tasa, requisitos de privacidad y migraciones cuando un modelo se retira. La calidad no puede evaluarse leyendo una tabla comercial: todos los modelos se ejecutarán sobre los mismos textos, candidatos y métricas.

                Las celdas siguientes realizan llamadas reales. Leen `OPENAI_API_KEY`, `COHERE_API_KEY` y `GEMINI_API_KEY` desde `.env`; no contienen interruptores adicionales. Se calculan embeddings de los 336 productos, las 12 consultas originales y las 8 paráfrasis. Cada proveedor recibe el rol correcto de consulta o documento cuando su contrato lo permite.

                El coste total es pequeño para esta muestra, pero no es cero. En un catálogo real los documentos se embeben por lotes y se persisten; no deben recalcularse en cada ejecución analítica. También deben registrarse latencia, modelo, dimensión y fecha, porque una comparación sin configuración no es reproducible.
                """
            ),
            code(
                """
                from vector_search_session.model_catalog import load_model_catalog

                model_catalog = load_model_catalog()
                provider_model_ids = {
                    "openai-text-embedding-3-small",
                    "openai-text-embedding-3-large",
                    "cohere-embed-v4",
                    "google-gemini-embedding-001",
                    "google-gemini-embedding-2",
                }
                provider_rows = [
                    model.as_table_record()
                    for model in model_catalog.models
                    if model.identifier in provider_model_ids
                ]
                pd.DataFrame(provider_rows)[
                    ["modelo", "estado", "modalidades", "contexto_tokens", "dimensiones"]
                ]
                """
            ),
            code(
                """
                from time import perf_counter

                provider_document_texts = products["searchable_text"].tolist()
                provider_original_texts = unique_queries["query"].tolist()
                provider_semantic_texts = semantic_queries["semantic_query"].tolist()
                api_timings = []
                """,
                tags=("requires-api-key",),
            ),
            markdown(
                r"""
                ### OpenAI: `text-embedding-3-small` y `text-embedding-3-large`

                Ambos modelos reciben texto mediante `client.embeddings.create`. `text-embedding-3-small` entrega 1.536 dimensiones por defecto y prioriza coste y velocidad. `text-embedding-3-large` entrega 3.072 y dispone de mayor capacidad. El parámetro `dimensions` permite solicitar una representación más corta entrenada con propiedades Matryoshka.

                OpenAI devuelve vectores normalizados, por lo que coseno y producto escalar producen el mismo ranking. Aun así, la función normaliza de nuevo de forma defensiva después de validar la forma. No se mezclan vectores de `small` y `large`, aunque se solicite la misma dimensión: compartir shape no significa compartir espacio.

                Para hacer comparable el coste de almacenamiento se solicitan 512 dimensiones en ambos modelos. Esta decisión no presupone que sea el mejor punto; forma parte del experimento y queda registrada en el nombre del sistema.
                """
            ),
            code(
                """
                from openai import OpenAI

                openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

                def embed_openai_batch(texts: list[str], model: str) -> np.ndarray:
                    response = openai_client.embeddings.create(
                        model=model,
                        input=texts,
                        dimensions=512,
                        encoding_format="float",
                    )
                    vectors = np.asarray(
                        [item.embedding for item in response.data], dtype=np.float32
                    )
                    return safe_l2_normalize(vectors, axis=1)
                """,
                tags=("requires-api-key",),
            ),
            code(
                """
                def encode_in_batches(
                    texts: list[str], encoder, batch_size: int
                ) -> np.ndarray:
                    batches = []
                    for batch_start in range(0, len(texts), batch_size):
                        text_batch = texts[batch_start : batch_start + batch_size]
                        batches.append(encoder(text_batch))
                    return np.vstack(batches)
                """,
                tags=("requires-api-key",),
            ),
            code(
                """
                started_at = perf_counter()
                openai_small_products = encode_in_batches(
                    provider_document_texts,
                    lambda batch: embed_openai_batch(batch, "text-embedding-3-small"),
                    128,
                )
                openai_small_original = embed_openai_batch(
                    provider_original_texts, "text-embedding-3-small"
                )
                openai_small_semantic = embed_openai_batch(
                    provider_semantic_texts, "text-embedding-3-small"
                )
                api_timings.append(
                    ("OpenAI small · 512d", perf_counter() - started_at)
                )
                """,
                tags=("requires-api-key",),
            ),
            code(
                """
                started_at = perf_counter()
                openai_large_products = encode_in_batches(
                    provider_document_texts,
                    lambda batch: embed_openai_batch(batch, "text-embedding-3-large"),
                    128,
                )
                openai_large_original = embed_openai_batch(
                    provider_original_texts, "text-embedding-3-large"
                )
                openai_large_semantic = embed_openai_batch(
                    provider_semantic_texts, "text-embedding-3-large"
                )
                api_timings.append(
                    ("OpenAI large · 512d", perf_counter() - started_at)
                )
                """,
                tags=("requires-api-key",),
            ),
            markdown(
                r"""
                ### Cohere `embed-v4.0`: rol explícito y entrada multimodal

                Cohere obliga a declarar `input_type`. Para recuperación, los documentos utilizan `search_document` y las consultas `search_query`. Esta distinción permite entrenar un espacio asimétrico: una ficha extensa y una consulta breve no tienen la misma distribución, aunque deban ser comparables.

                `embed-v4.0` admite salidas de 256, 512, 1.024 o 1.536 dimensiones y entradas de texto, imagen o combinaciones. Aquí se eligen 1.024 dimensiones y embeddings `float`. La API limita cada llamada a 96 entradas, por lo que el catálogo se divide en lotes.

                El tipo de entrada debe considerarse parte de la versión del índice. Embeber documentos como `search_query` no genera un error de shape: genera vectores válidos en una configuración equivocada, un fallo mucho más difícil de detectar.
                """
            ),
            code(
                """
                import cohere

                cohere_client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])

                def embed_cohere_batch(texts: list[str], input_type: str) -> np.ndarray:
                    response = cohere_client.embed(
                        texts=texts,
                        model="embed-v4.0",
                        input_type=input_type,
                        embedding_types=["float"],
                        output_dimension=1_024,
                    )
                    vectors = np.asarray(response.embeddings.float_, dtype=np.float32)
                    return safe_l2_normalize(vectors, axis=1)
                """,
                tags=("requires-api-key",),
            ),
            code(
                """
                started_at = perf_counter()
                cohere_products = encode_in_batches(
                    provider_document_texts,
                    lambda batch: embed_cohere_batch(batch, "search_document"),
                    96,
                )
                cohere_original = embed_cohere_batch(
                    provider_original_texts, "search_query"
                )
                cohere_semantic = embed_cohere_batch(
                    provider_semantic_texts, "search_query"
                )
                api_timings.append(("Cohere Embed 4 · 1024d", perf_counter() - started_at))
                """,
                tags=("requires-api-key",),
            ),
            markdown(
                r"""
                ### Google: `gemini-embedding-001` frente a `gemini-embedding-2`

                Los dos modelos no son versiones intercambiables del mismo espacio. `gemini-embedding-001` es textual, acepta `task_type` como `RETRIEVAL_DOCUMENT` o `RETRIEVAL_QUERY` y genera un vector independiente para cada string de una lista. `gemini-embedding-2` es multimodal, amplía el contexto y sustituye `task_type` por instrucciones escritas en la entrada.

                Con Embedding 2 hay un detalle especialmente importante: si se envían varias partes dentro de un único `Content`, el modelo puede agregarlas en un solo embedding multimodal. Para obtener un vector por producto se construye un objeto `Content` independiente para cada texto. Los documentos reciben una plantilla con título y texto; las consultas reciben una instrucción de recuperación.

                Ambos modelos permiten controlar la dimensión de salida, pero no comparten espacio. `gemini-embedding-2` normaliza automáticamente las dimensiones reducidas; `gemini-embedding-001` exige normalizarlas manualmente cuando son inferiores a la salida completa. Migrar entre ambos obliga a recalcular documentos y consultas. Actualizar solo la consulta produciría vectores formalmente válidos que no pueden compararse con el índice anterior.
                """
            ),
            code(
                """
                from google import genai
                from google.genai import types

                google_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

                def google_values(response) -> np.ndarray:
                    vectors = np.asarray(
                        [item.values for item in response.embeddings], dtype=np.float32
                    )
                    return safe_l2_normalize(vectors, axis=1)
                """,
                tags=("requires-api-key",),
            ),
            code(
                """
                def embed_google_001_batch(
                    texts: list[str], task_type: str
                ) -> np.ndarray:
                    response = google_client.models.embed_content(
                        model="gemini-embedding-001",
                        contents=texts,
                        config=types.EmbedContentConfig(
                            task_type=task_type,
                            output_dimensionality=768,
                        ),
                    )
                    return google_values(response)
                """,
                tags=("requires-api-key",),
            ),
            code(
                """
                google_001_available = True
                try:
                    started_at = perf_counter()
                    google_001_products = encode_in_batches(
                        provider_document_texts,
                        lambda batch: embed_google_001_batch(batch, "RETRIEVAL_DOCUMENT"),
                        64,
                    )
                    google_001_original = embed_google_001_batch(
                        provider_original_texts, "RETRIEVAL_QUERY"
                    )
                    google_001_semantic = embed_google_001_batch(
                        provider_semantic_texts, "RETRIEVAL_QUERY"
                    )
                    api_timings.append(("Gemini 001 · 768d", perf_counter() - started_at))
                except Exception as legacy_error:
                    google_001_available = False
                    print(f"gemini-embedding-001 no disponible: {legacy_error}")
                """,
                tags=("requires-api-key",),
            ),
            code(
                """
                def embed_google_2_batch(
                    texts: list[str], input_type: str
                ) -> np.ndarray:
                    prefix = (
                        "task: search result | query: "
                        if input_type == "query"
                        else "title: product | text: "
                    )
                    contents = [
                        types.Content(parts=[types.Part(text=prefix + text)])
                        for text in texts
                    ]
                    response = google_client.models.embed_content(
                        model="gemini-embedding-2",
                        contents=contents,
                        config=types.EmbedContentConfig(output_dimensionality=768),
                    )
                    return google_values(response)
                """,
                tags=("requires-api-key",),
            ),
            code(
                """
                started_at = perf_counter()
                google_2_products = encode_in_batches(
                    provider_document_texts,
                    lambda batch: embed_google_2_batch(batch, "document"),
                    64,
                )
                google_2_original = embed_google_2_batch(
                    provider_original_texts, "query"
                )
                google_2_semantic = embed_google_2_batch(
                    provider_semantic_texts, "query"
                )
                api_timings.append(("Gemini 2 · 768d", perf_counter() - started_at))
                """,
                tags=("requires-api-key",),
            ),
            markdown(
                r"""
                ## 6.4. Open-weight no significa una única familia

                E5 es solo un punto del mapa. BGE-M3 puede producir representaciones dense, sparse aprendidas y late-interaction desde un mismo modelo. Qwen3-Embedding admite instrucciones y dimensiones Matryoshka. EmbeddingGemma prioriza despliegue compacto. La elección depende del idioma, longitud, licencia, hardware, modalidad y patrón de consulta.

                Operar pesos propios evita enviar el corpus a una API y permite controlar versiones, cuantización y batching. A cambio, hay que aprovisionar memoria, servir el modelo, gestionar concurrencia, medir colas, desplegar actualizaciones y vigilar que tokenizer y checkpoint coincidan. `Open-weight` describe acceso al artefacto; no significa coste operativo cero ni licencia sin restricciones.

                La comparación correcta no enfrenta `API` contra `open source` como etiquetas abstractas. Enfrenta configuraciones concretas: modelo, dimensión, plantilla, normalización, hardware, coste y calidad sobre las mismas consultas.
                """
            ),
        ]
    )

    cells.extend(
        [
            markdown(
                r"""
                # 7. Más allá de un único vector denso

                Un solo vector comprime toda la secuencia. Esa compresión permite recuperar con enorme eficiencia, pero puede borrar interacciones finas: una medida concreta, una negación o la correspondencia entre varios atributos.

                ## 7.1. Sparse aprendido y SPLADE

                **SPLADE** utiliza un Transformer para producir un vector sparse sobre el vocabulario. Sus dimensiones siguen siendo términos interpretables, pero el modelo puede expandir conceptos que no aparecen literalmente y aprender sus pesos. Mantiene compatibilidad con índices invertidos a cambio de una representación aprendida más costosa.

                ## 7.2. Recuperación híbrida

                Un sistema **híbrido** conserva dos rankings, por ejemplo TF-IDF/BM25 y un dense. Reciprocal Rank Fusion combina posiciones mediante $\sum_r 1/(k+r)$ y evita asumir que un score léxico y un coseno comparten escala. Otra opción consiste en aprender una fusión con features, pero necesita datos suficientes y una validación cuidadosa.

                ## 7.3. Late interaction y ColBERT

                **ColBERT** conserva un vector por token. Para cada token de consulta busca el token documental más parecido y suma los máximos: $\operatorname{MaxSim}(Q,D)=\sum_i\max_j \mathbf{q}_i^\top\mathbf{d}_j$. Esta late interaction retiene coincidencias detalladas sin ejecutar un cross-encoder completo sobre todo el catálogo, aunque multiplica almacenamiento y cálculo.
                """
            ),
            code(
                """
                query_tokens = ["asiento", "trabajo", "espalda"]
                product_tokens = ["silla", "oficina", "lumbar", "roja"]
                token_similarities = np.array(
                    [
                        [0.88, 0.22, 0.34, 0.05],
                        [0.18, 0.91, 0.25, 0.03],
                        [0.30, 0.31, 0.86, 0.02],
                    ]
                )
                maxsim_score = token_similarities.max(axis=1).sum()
                """
            ),
            code(
                """
                maxsim_figure = px.imshow(
                    token_similarities,
                    x=product_tokens,
                    y=query_tokens,
                    text_auto=".2f",
                    color_continuous_scale="Blues",
                    zmin=0,
                    zmax=1,
                )
                maxsim_figure.update_layout(
                    title=f"Late interaction: MaxSim = {maxsim_score:.2f}"
                )
                maxsim_figure.show()
                """
            ),
            markdown(
                r"""
                ## 7.4. Embeddings multimodales

                Hasta ahora toda la información ha sido textual. Un embedding **multimodal** amplía el espacio para representar imágenes, audio, vídeo u otros contenidos junto al texto. Esto permite que una consulta escrita recupere fotografías de producto o que una imagen encuentre fichas visualmente relacionadas.

                CLIP es uno de los ejemplos más conocidos: entrena un encoder visual y otro textual para acercar cada imagen a su descripción correcta y separarla de las demás. Entre los servicios gestionados, Cohere `embed-v4.0` admite texto e imágenes, mientras que `gemini-embedding-2` incorpora texto, imagen, audio, vídeo y PDF dentro de un espacio compartido.

                En un marketplace, esta capacidad aporta valor cuando el color, el patrón, la forma o el acabado aparecen en la fotografía pero no en la descripción. Añadir imágenes no garantiza una mejora: modifica el contrato del índice y obliga a evaluar consultas en las que la señal visual sea realmente necesaria.
                """
            ),
        ]
    )

    cells.extend(
        [
            markdown(
                r"""
                # 8. Evaluación del buscador

                Para cada consulta ESCI solo se conocen juicios sobre un conjunto de candidatos. La evaluación se realiza dentro de ese conjunto. Buscar sobre los 336 productos es útil para inspección, pero no permite declarar irrelevante cualquier producto sin etiqueta. Confundir `no juzgado` con `irrelevante` sesga las métricas.

                ## 8.1. Recall@k: comprobar la cobertura de la recuperación

                La primera pregunta es si los productos relevantes aparecen dentro de los primeros $k$ candidatos. **Recall@k** mide qué proporción del conjunto relevante ha sobrevivido a esa primera selección:

                $$
                \operatorname{Recall@k}
                =\frac{|\operatorname{Rel}\cap\operatorname{Top}_k|}
                {|\operatorname{Rel}|}.
                $$

                En este ejercicio consideraremos relevantes las etiquetas `Exact` y `Substitute`. `Complement` no resuelve por sí solo la necesidad, aunque nDCG conservará su pequeña ganancia graduada. Esta decisión debe hacerse explícita: cambiar qué etiquetas forman $\operatorname{Rel}$ cambia la pregunta que responde la métrica.

                Recall@k resulta decisivo en una arquitectura de dos etapas. Si el recuperador deja fuera un producto, ningún reranker podrá devolverlo después. La métrica no valora el orden dentro del top-$k$; solo comprueba que la primera etapa ha proporcionado cobertura suficiente.

                ## 8.2. nDCG: valorar el orden y la relevancia graduada

                Una vez comprobada la cobertura, necesitamos medir cómo se ordenan los candidatos. ESCI asigna ganancias `E=1`, `S=0.1`, `C=0.01`, `I=0`. DCG descuenta logarítmicamente la ganancia según la posición:

                $$DCG=\sum_{r=1}^{n}\frac{gain_r}{\log_2(r+1)}.$$

                El ranking ideal coloca primero todos los Exact, después Substitute, Complement e Irrelevant. Dividir por su DCG produce nDCG entre 0 y 1. La métrica premia especialmente los primeros puestos, donde se concentra la atención del usuario, y permite comparar consultas con distinto número de candidatos.

                Se calculará una puntuación por consulta y después una media macro: cada consulta pesa lo mismo. En producción podrían añadirse pesos por frecuencia o valor, pero entonces la métrica respondería a otra pregunta.
                """
            ),
            code(
                """
                def esci_ndcg(labels: pd.Series, scores: np.ndarray) -> float:
                    gains = labels.map(ESCI_GAINS).to_numpy(dtype=float)
                    ranked_gains = gains[np.argsort(-scores, kind="stable")]
                    discounts = np.log2(np.arange(2, len(gains) + 2))
                    dcg = np.sum(ranked_gains / discounts)
                    ideal_dcg = np.sum(np.sort(gains)[::-1] / discounts)
                    return float(dcg / ideal_dcg) if ideal_dcg else 0.0


                def esci_recall_at_k(
                    labels: pd.Series,
                    scores: np.ndarray,
                    k: int = 10,
                ) -> float:
                    relevant = labels.isin(["E", "S"]).to_numpy()
                    relevant_total = int(relevant.sum())
                    if relevant_total == 0:
                        return 0.0
                    top_k = np.argsort(-scores, kind="stable")[:k]
                    return float(relevant[top_k].sum() / relevant_total)
                """
            ),
            code(
                """
                product_position = {
                    product_id: position
                    for position, product_id in enumerate(products["product_id"])
                }
                original_query_position = {
                    int(query_id): position
                    for position, query_id in enumerate(unique_queries["query_id"])
                }
                semantic_query_position = {
                    int(query_id): position
                    for position, query_id in enumerate(semantic_queries["query_id"])
                }
                """
            ),
            code(
                """
                def judged_candidates(query_id: int) -> tuple[pd.DataFrame, np.ndarray]:
                    query_group = judgments.loc[judgments["query_id"] == query_id]
                    positions = np.array(
                        [product_position[item] for item in query_group["product_id"]]
                    )
                    return query_group, positions
                """
            ),
            code(
                """
                def evaluation_record(
                    model_name: str,
                    query_id: int,
                    query_type: str,
                    labels: pd.Series,
                    scores: np.ndarray,
                ) -> dict[str, object]:
                    return {
                        "modelo": model_name,
                        "query_id": query_id,
                        "tipo": query_type,
                        "Recall@10": esci_recall_at_k(labels, scores, k=10),
                        "nDCG": esci_ndcg(labels, scores),
                    }
                """
            ),
            code(
                """
                def evaluate_dense_view(
                    model_name: str,
                    document_embeddings: np.ndarray,
                    query_ids: np.ndarray,
                    query_embeddings: np.ndarray,
                    query_type: str,
                ) -> list[dict[str, object]]:
                    rows = []
                    for query_id, query_vector in zip(
                        query_ids, query_embeddings, strict=True
                    ):
                        query_group, positions = judged_candidates(int(query_id))
                        scores = document_embeddings[positions] @ query_vector
                        rows.append(
                            evaluation_record(
                                model_name,
                                int(query_id),
                                query_type,
                                query_group["esci_label"],
                                scores,
                            )
                        )
                    return rows
                """
            ),
            code(
                """
                def evaluate_dense_system(
                    model_name: str,
                    document_embeddings: np.ndarray,
                    original_embeddings: np.ndarray,
                    semantic_embeddings: np.ndarray,
                ) -> list[dict[str, object]]:
                    original_rows = evaluate_dense_view(
                        model_name,
                        document_embeddings,
                        unique_queries["query_id"].to_numpy(),
                        original_embeddings,
                        "original",
                    )
                    return original_rows + evaluate_dense_view(
                        model_name, document_embeddings,
                        semantic_queries["query_id"].to_numpy(),
                        semantic_embeddings, "paráfrasis",
                    )
                """
            ),
            code(
                """
                def evaluate_sparse_view(
                    query_frame: pd.DataFrame,
                    text_column: str,
                    query_type: str,
                ) -> list[dict[str, object]]:
                    rows = []
                    for query_row in query_frame.itertuples(index=False):
                        query_id = int(query_row.query_id)
                        query_group, positions = judged_candidates(query_id)
                        query_text = getattr(query_row, text_column)
                        query_sparse = tfidf_vectorizer.transform([query_text])
                        scores = (
                            tfidf_matrix[positions] @ query_sparse.T
                        ).toarray().ravel()
                        rows.append(
                            evaluation_record(
                                "TF-IDF",
                                query_id,
                                query_type,
                                query_group["esci_label"],
                                scores,
                            )
                        )
                    return rows
                """
            ),
            code(
                """
                def evaluate_tfidf_system() -> list[dict[str, object]]:
                    original_rows = evaluate_sparse_view(
                        unique_queries, "query", "original"
                    )
                    semantic_rows = evaluate_sparse_view(
                        semantic_queries, "semantic_query", "paráfrasis"
                    )
                    return original_rows + semantic_rows
                """
            ),
            code(
                """
                evaluation_rows = evaluate_tfidf_system()
                evaluation_rows.extend(
                    evaluate_dense_system(
                        "multilingual-e5-small",
                        e5_product_embeddings,
                        e5_query_embeddings,
                        e5_semantic_embeddings,
                    )
                )
                local_evaluation = pd.DataFrame(evaluation_rows)
                """
            ),
            code(
                """
                local_summary = (
                    local_evaluation.groupby(["modelo", "tipo"], as_index=False)[
                        ["Recall@10", "nDCG"]
                    ]
                    .mean()
                    .sort_values(["tipo", "nDCG"], ascending=[True, False])
                )
                local_summary
                """
            ),
            code(
                """
                semantic_comparison = (
                    local_evaluation.loc[local_evaluation["tipo"] == "paráfrasis"]
                    .pivot(index="query_id", columns="modelo", values="nDCG")
                    .reset_index()
                    .merge(
                        semantic_queries[["query_id", "semantic_query"]],
                        on="query_id",
                    )
                )
                semantic_comparison["delta_E5"] = (
                    semantic_comparison["multilingual-e5-small"]
                    - semantic_comparison["TF-IDF"]
                )
                """
            ),
            code(
                """
                semantic_figure = px.bar(
                    semantic_comparison.sort_values("delta_E5"),
                    x="delta_E5",
                    y="semantic_query",
                    orientation="h",
                    color="delta_E5",
                    color_continuous_scale="RdBu",
                    color_continuous_midpoint=0,
                )
                semantic_figure.update_layout(
                    title="Cambio de nDCG al sustituir TF-IDF por E5",
                    xaxis_title="nDCG(E5) - nDCG(TF-IDF)",
                    yaxis_title="Paráfrasis",
                    height=620,
                )
                semantic_figure.show()
                """
            ),
            markdown(
                r"""
                La comparación local separa dos fenómenos. En consultas originales, TF-IDF puede ser muy competitivo porque los usuarios del dataset suelen escribir nombres de categoría y atributos. En las paráfrasis, el solapamiento superficial se reduce de forma deliberada y E5 debería perder menos calidad.

                El gráfico impide reducir la conclusión a `dense gana`. En esta muestra E5 mejora con claridad las necesidades expresadas como problema de espalda y cambio de unidad, pero pierde en varias paráfrasis de compatibilidad o restricción. Puede haber ambigüedad en la paráfrasis, información insuficiente en la ficha o una frontera que el encoder no aprendió bien. Cada barra negativa es un caso que debe inspeccionarse antes de desplegar.

                Si el dense mejora las paráfrasis pero empeora medidas, negaciones o modelos, el siguiente experimento razonable es híbrido. Si una API mejora unas milésimas a cambio de multiplicar coste y latencia, puede no justificar la migración. Si un modelo grande no supera a uno pequeño en este slice, la dimensión extra no aporta valor demostrado.

                Las celdas siguientes incorporan los cinco modelos API al mismo DataFrame. No se comparan sus cosenos crudos; se compara el ranking producido por cada espacio mediante nDCG.
                """
            ),
            code(
                """
                api_evaluation_rows = []
                api_systems = [
                    ("OpenAI small · 512d", openai_small_products, openai_small_original, openai_small_semantic),
                    ("OpenAI large · 512d", openai_large_products, openai_large_original, openai_large_semantic),
                    ("Cohere Embed 4 · 1024d", cohere_products, cohere_original, cohere_semantic),
                    ("Gemini 2 · 768d", google_2_products, google_2_original, google_2_semantic),
                ]
                if google_001_available:
                    api_systems.append(
                        ("Gemini 001 · 768d", google_001_products, google_001_original, google_001_semantic)
                    )
                for system_name, document_vectors, original_vectors, semantic_vectors in api_systems:
                    api_evaluation_rows.extend(
                        evaluate_dense_system(
                            system_name,
                            document_vectors,
                            original_vectors,
                            semantic_vectors,
                        )
                    )
                """,
                tags=("requires-api-key",),
            ),
            code(
                """
                full_evaluation = pd.concat(
                    [local_evaluation, pd.DataFrame(api_evaluation_rows)],
                    ignore_index=True,
                )
                full_summary = (
                    full_evaluation.groupby(["modelo", "tipo"], as_index=False)["nDCG"]
                    .mean()
                    .sort_values(["tipo", "nDCG"], ascending=[True, False])
                )
                full_summary
                """,
                tags=("requires-api-key",),
            ),
            code(
                """
                comparison_figure = px.bar(
                    full_summary,
                    x="modelo",
                    y="nDCG",
                    color="tipo",
                    barmode="group",
                    text_auto=".3f",
                )
                comparison_figure.update_layout(
                    title="Calidad en consultas originales y paráfrasis",
                    xaxis_tickangle=-25,
                    height=580,
                )
                comparison_figure.show()
                """,
                tags=("requires-api-key",),
            ),
            code(
                """
                latency_frame = pd.DataFrame(
                    api_timings, columns=["modelo", "segundos_embedding_muestra"]
                )
                latency_frame["dimension"] = latency_frame["modelo"].str.extract(
                    r"(\\d+)d"
                ).astype(int)
                latency_frame
                """,
                tags=("requires-api-key",),
            ),
            markdown(
                r"""
                ## 8.3. Latencia, memoria y coste

                Los tiempos anteriores son medidas de pared sobre una ejecución pequeña. Incluyen serialización, red, colas del proveedor y batching; no son un benchmark estable ni deben utilizarse como SLA. Una comparación operativa repetiría ejecuciones, separaría indexación de consulta, registraría percentiles y controlaría región, concurrencia y reintentos.

                Tampoco se fijan precios en el notebook porque cambian con el tiempo y con el contrato. El coste debe calcularse con el uso medido y la tarifa vigente en el momento de la decisión. Lo reproducible aquí es el número de textos, la configuración del modelo y el tiempo observado, no una estimación monetaria congelada.
                """
            ),
            markdown(
                r"""
                ## 8.4. Análisis por consulta: la media no explica el fallo

                Una media macro puede ocultar que un sistema gana ligeramente en siete consultas y falla de forma grave en una consulta crítica. Por eso se inspecciona el delta por `query_id` y el ranking concreto.

                En el conjunto original, `cámaras bridge baratas` tiende a favorecer a TF-IDF: contiene una categoría precisa y el dense puede acercar accesorios o cámaras relacionadas. `sillas oficina ergonomicas` y su paráfrasis permiten observar el caso contrario. La representación semántica puede conectar el problema de espalda y la jornada larga con atributos ergonómicos que no aparecen literalmente en la consulta.

                El análisis de errores debe convertir observaciones en nuevos slices: medidas y unidades, negaciones, compatibilidad, marcas, consultas por problema, complementos y resultados sin juicio. Un benchmark propio mejora cuando cada fallo recurrente se transforma en una prueba estable.
                """
            ),
            code(
                """
                local_pivot = local_evaluation.pivot_table(
                    index=["query_id", "tipo"],
                    columns="modelo",
                    values="nDCG",
                ).reset_index()
                local_pivot["delta_E5"] = (
                    local_pivot["multilingual-e5-small"] - local_pivot["TF-IDF"]
                )
                local_pivot.sort_values("delta_E5", ascending=False)
                """
            ),
            markdown(
                r"""
                # 9. Selección de la arquitectura

                MTEB y MMTEB son útiles para reducir una lista de cientos de modelos. Agregan tareas, idiomas y dominios bajo protocolos comunes. No sustituyen la evaluación del buscador: el catálogo, las consultas, la relevancia, la longitud, el idioma y la infraestructura pueden diferir de los benchmarks públicos.

                Una decisión defendible combina al menos:

                1. calidad media y por slices en juicios propios;
                2. coste de indexación y coste por consulta;
                3. latencia de codificación y recuperación;
                4. memoria del vector y del índice;
                5. privacidad, licencia y dependencia del proveedor;
                6. estrategia de versionado y migración.

                Para este marketplace, la hipótesis de salida más razonable no es eliminar TF-IDF. Es mantener una señal léxica para referencias y restricciones, añadir el mejor dense validado para consultas por necesidad y medir una fusión híbrida. Solo después tendría sentido experimentar con reranking, late interaction o multimodalidad.

                El aprendizaje central puede resumirse con una cadena causal: **el texto de entrada determina la representación; el entrenamiento determina la geometría; la métrica convierte esa geometría en ranking; los juicios de negocio determinan si el ranking es bueno**.
                """
            ),
        ]
    )

    return cells


def order_narrative_sections(
    cells: list[nbformat.NotebookNode],
) -> list[nbformat.NotebookNode]:
    """Place each practical block in the same conceptual order as the memory."""

    def markdown_index(prefix: str) -> int:
        return next(
            index
            for index, cell in enumerate(cells)
            if cell.cell_type == "markdown" and cell.source.startswith(prefix)
        )

    matryoshka_start = markdown_index("## 6.3. Matryoshka")
    api_start = markdown_index("## 6.2. Modelos de embeddings mediante API")

    matryoshka_block = cells[matryoshka_start:api_start]
    del cells[matryoshka_start:api_start]

    open_weight_start = markdown_index("## 6.4. Open-weight")
    cells[open_weight_start:open_weight_start] = matryoshka_block
    return cells


def add_execution_bridges(
    cells: list[nbformat.NotebookNode],
) -> list[nbformat.NotebookNode]:
    """Connect every conceptual explanation with the experiment that follows."""

    bridges = {
        "## 1. El problema": (
            "Empezaremos preparando el entorno de ejecución y cargando la muestra ESCI. "
            "Después comprobaremos sus consultas, productos y paráfrasis antes de construir "
            "ninguna representación; así sabremos exactamente qué datos va a recibir cada modelo."
        ),
        "La tabla anterior hace visible": (
            "Antes de abandonar los datos, resumiremos ahora cuántos juicios hay de cada clase. "
            "La distribución resultante nos permitirá comprobar si la evaluación estará dominada "
            "por alguna etiqueta y dará contexto a las métricas del último bloque."
        ),
        "### 1.2. Qué representa": (
            "Vamos a escoger una ficha concreta, mostrar sus campos originales y compararlos con "
            "el texto final que recibirá el encoder. El objetivo es hacer visible una transformación "
            "que, de otro modo, quedaría escondida dentro de una función auxiliar."
        ),
        "# 2. La geometría": (
            "Para observar estas decisiones sin el ruido de cientos de coordenadas, construiremos "
            "primero una consulta y varios productos en dos dimensiones. Los dibujaremos y "
            "utilizaremos ese mismo ejemplo en las comparaciones geométricas posteriores."
        ),
        "## 2.1. Coseno": (
            "Aplicaremos ahora las tres métricas a los mismos candidatos. Mantener fijos los "
            "vectores nos permitirá atribuir cualquier cambio de ranking exclusivamente a la "
            "función de comparación."
        ),
        "## 2.2. La norma": (
            "Vamos a aislar este efecto multiplicando la consulta por cinco y repitiendo las "
            "comparaciones. La dirección permanecerá intacta; únicamente cambiará la magnitud, "
            "de modo que podremos ver qué métricas reaccionan a ella."
        ),
        "## 2.3. Normalización": (
            "Normalizaremos a continuación la consulta y los candidatos, recalcularemos coseno, "
            "producto escalar y L2, y verificaremos numéricamente la equivalencia de sus rankings. "
            "La aserción final hará que el notebook falle si esa propiedad deja de cumplirse."
        ),
        "## 2.4. Dimensionalidad": (
            "El siguiente experimento no utilizará embeddings aprendidos, sino nubes aleatorias "
            "de dimensión creciente. Mediremos la relación entre la distancia mínima y la máxima "
            "para visualizar la concentración sin confundirla con la calidad de un encoder real."
        ),
        "# 3. Representaciones dispersas": (
            "Construiremos la matriz Bag-of-Words del catálogo y mediremos su forma, su vocabulario "
            "y el porcentaje real de ceros. De esta forma, `sparse` dejará de ser una etiqueta "
            "abstracta y se convertirá en una propiedad observable de nuestros datos."
        ),
        "## 3.2. TF-IDF": (
            "Sustituiremos ahora los conteos por pesos TF-IDF aprendidos únicamente sobre el "
            "catálogo. Inspeccionaremos los términos con mayor IDF para comprobar qué palabras "
            "considera especialmente discriminativas esta colección concreta."
        ),
        "## 3.3. La consulta": (
            "La comparación práctica utilizará una consulta literal y su paráfrasis semántica. "
            "Ambas atravesarán el mismo `transform` y recuperarán productos sobre el mismo índice, "
            "por lo que la única diferencia será la forma de expresar la intención."
        ),
        "# 4. Primeras representaciones": (
            "Entrenaremos un pequeño Skip-gram sobre las fichas del catálogo y consultaremos los "
            "vecinos de `silla`. El corpus es deliberadamente reducido: buscamos observar el "
            "mecanismo y sus límites, no producir un embedding competitivo."
        ),
        "El entrenamiento anterior utiliza": (
            "Como GloVe no se entrenará sobre esta muestra mínima, situaremos Word2Vec y GloVe en "
            "una tabla comparativa antes de avanzar. La tabla fijará qué señal utiliza cada uno y "
            "qué limitaciones comparten pese a sus objetivos diferentes."
        ),
        "## 4.3. FastText": (
            "Entrenaremos FastText con el mismo corpus y pediremos un vector para `ergonomikas`, "
            "una forma ausente del vocabulario. Compararlo con Word2Vec permitirá atribuir el "
            "resultado a la composición por subpalabras y no a diferencias en los datos."
        ),
        "## 5.2. Sentence-BERT": (
            "Para convertir la diferencia arquitectónica en una consecuencia operativa, "
            "estimaremos cuántas evaluaciones requiere un cross-encoder al crecer el catálogo y "
            "cuántos encodings necesita un bi-encoder con documentos precalculados."
        ),
        "# 6. Modelos modernos": (
            "Cargaremos ahora los tres archivos de embeddings E5 —productos, consultas originales "
            "y paráfrasis— junto con sus metadatos. Antes de buscar, validaremos IDs, dimensiones y "
            "normas para impedir que un índice desalineado produzca resultados plausibles pero falsos."
        ),
        "### La prueba decisiva": (
            "Recuperaremos la misma pareja de consultas con TF-IDF y con E5, y mostraremos ambos "
            "rankings uno junto a otro. Los títulos permitirán juzgar qué sistema conserva mejor la "
            "intención cuando desaparece el vocabulario literal."
        ),
        "## 6.2. Modelos de embeddings": (
            "Antes de llamar a ningún proveedor, cargaremos el catálogo versionado de modelos y "
            "crearemos una estructura común para registrar embeddings, dimensión, tiempo y estado. "
            "Así todas las APIs desembocarán en el mismo protocolo de evaluación."
        ),
        "### OpenAI": (
            "Crearemos el cliente de OpenAI y una función de batching reutilizable. Después "
            "calcularemos por separado documentos, consultas originales y paráfrasis para `small` "
            "y `large`, conservando la configuración exacta de cada ejecución."
        ),
        "### Cohere": (
            "La implementación enviará el catálogo como `search_document` y las dos colecciones de "
            "consultas como `search_query`. También dividirá las entradas en lotes para respetar el "
            "límite del endpoint y registrará el tiempo total de indexación."
        ),
        "### Google": (
            "Construiremos dos rutas distintas porque los contratos de `001` y `2` no son "
            "intercambiables. Cada una generará documentos y consultas con su rol correspondiente, "
            "y cualquier indisponibilidad quedará registrada sin ocultar la llamada real."
        ),
        "## 6.3. Matryoshka": (
            "Calcularemos ahora el almacenamiento bruto de un millón de vectores para varias "
            "dimensiones. El gráfico no elegirá el tamaño por nosotros, pero hará tangible el coste "
            "que después deberá confrontarse con recall y nDCG."
        ),
        "# 7. Más allá": (
            "Para observar la late interaction sin implementar todavía un índice ColBERT completo, "
            "construiremos una matriz pequeña de similitudes token a token. Calcularemos MaxSim y "
            "veremos qué correspondencia aporta cada palabra de la consulta."
        ),
        "# 8. Evaluación": (
            "Implementaremos primero recall@10 y nDCG como funciones independientes. Después "
            "crearemos un registro común por modelo, consulta y tipo de entrada, de manera que la "
            "misma evaluación sirva para TF-IDF, E5 y los proveedores gestionados."
        ),
        "La comparación local separa": (
            "Incorporaremos ahora los resultados disponibles de las APIs al mismo DataFrame. "
            "Las llamadas que no hayan podido ejecutarse no se convertirán en ceros: simplemente "
            "quedarán fuera de la comparación para no confundir indisponibilidad con mala calidad."
        ),
        "## 8.4. Análisis": (
            "Cerraremos la evaluación construyendo una tabla por consulta con TF-IDF y E5. "
            "Ordenar por diferencia hará aparecer primero los fallos más graves y nos permitirá "
            "leer la media como el resumen de casos concretos, no como una conclusión aislada."
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
        cells=add_execution_bridges(order_narrative_sections(build_cells())),
        metadata={
            "kernelspec": {
                "display_name": "Python (BBDD Vectoriales · Sesión 1)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
            "case_study": "Búsqueda semántica de producto · Amazon ESCI",
        },
    )
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"Created {NOTEBOOK_PATH.name} with {len(notebook.cells)} cells")


if __name__ == "__main__":
    main()
