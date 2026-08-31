"""Notebook 00: business case, data exploration, and the lexical baseline."""

from __future__ import annotations

from nbformat import NotebookNode

from .common import code, markdown, setup_cells

FILENAME = "actividad_00_caso_datos_y_baseline.ipynb"


def build_cells() -> list[NotebookNode]:
    cells: list[NotebookNode] = [
        markdown(
            r"""
            # 00 · El caso, los datos y el baseline

            **Serie de I+D de la actividad Aurum Market.** Cada notebook justifica un
            bloque de decisiones; este cubre el **problema y el baseline** (bloque 1 de
            los criterios de evaluación): el contrato del sistema, qué contienen
            realmente los datos y una referencia léxica interpretable contra la que se
            medirá todo lo demás.

            | Notebook | Bloque del enunciado |
            |---|---|
            | **00 · Caso, datos y baseline** | §1 caso de negocio · §3.1 baseline |
            | 01 · Representación vectorial | §3.1 representación |
            | 02 · Índice y base de datos | §3.2 |
            | 03 · Recuperación y filtros | §3.3 |
            | 04 · Operaciones y duplicados | §4 |
            | 05 · Evaluación y análisis | §5 · §7 |

            **Requisito previo de la serie:** `make up && make embeddings && make
            pipeline` (los notebooks 01–05 leen artefactos del pipeline y consultan
            Qdrant). Este notebook 00 es la excepción: solo necesita los CSV del
            snapshot y `config/run_config.yaml`, así que puede ejecutarse sin Docker
            ni embeddings.

            Los notebooks están pensados para leerse **en orden**: cada concepto se
            explica la primera vez que aparece y los siguientes notebooks lo dan por
            sabido, así que la serie puede consumirse como fuente de conocimiento
            además de como justificación de la entrega.

            ## ¿Por qué una base de datos vectorial?

            La búsqueda clásica de un e-commerce es **léxica**: encuentra productos
            cuyos títulos comparten palabras con la consulta. Funciona hasta que la
            persona describe lo que necesita sin usar las palabras del catálogo
            («herramienta inalámbrica potente para perforar» no contiene «taladro»).
            La alternativa es representar el *significado* de cada texto como un
            **embedding**: un vector de números reales (1024 dimensiones en la
            configuración final) en un espacio donde la cercanía geométrica se
            corresponde con la cercanía semántica. Buscar deja de ser «¿qué títulos contienen estas palabras?» y
            pasa a ser «¿qué vectores están más cerca del vector de la consulta?».

            Una **base de datos vectorial** es el sistema que hace viable esa
            pregunta a escala: almacena los vectores junto a sus metadatos, los
            indexa para responder en milisegundos y mantiene el conjunto vivo
            (altas, cambios, bajas). Las tres capas — representación, índice y base
            de datos — son exactamente los tres notebooks centrales de la serie.

            ## El contrato del sistema

            Dos recorridos sobre una única interfaz normalizada (`DiscoveryService`):
            **descubrimiento** (consulta → top-k con score de semántica declarada,
            filtrable por marca dentro de la base de datos) y **control de altas**
            (ficha entrante → candidato más próximo → decisión de duplicado con
            `product_id` concreto). Fuera de alcance: RAG, generación y LLMs en tiempo
            de ejecución.
            """
        ),
        *setup_cells(),
        markdown(
            r"""
            ## Los datos, sin maquillar

            El catálogo son 15.000 fichas reales derivadas de ESCI (español). Antes de
            decidir nada conviene medir su suciedad: metadatos ausentes y un campo
            `text` con longitudes muy desiguales. Ninguna de estas anomalías se corrige
            a mano: son parte del problema.
            """
        ),
        code(
            r"""
            from aurum_discovery import load_catalog, load_manifest

            catalog = load_catalog()
            manifest = load_manifest()
            summary = pd.DataFrame(
                [
                    {"campo": column,
                     "vacíos": int((catalog[column] == "").sum()),
                     "% vacío": round(100 * (catalog[column] == "").mean(), 1)}
                    for column in ("brand", "color", "title", "text")
                ]
            )
            print(f"Registros: {len(catalog)} · snapshot: {manifest['snapshot_id']}")
            summary
            """
        ),
        code(
            r"""
            import plotly.express as px

            text_lengths = catalog["text"].str.len()
            figure = px.histogram(
                text_lengths, nbins=60,
                title="Longitud del campo text (caracteres): keyword-stuffing a la vista",
                labels={"value": "caracteres"},
            )
            figure.update_layout(showlegend=False)
            figure
            """
        ),
        markdown(
            r"""
            Buena parte de la cola que satura el tope de 3.000 caracteres es texto de
            posicionamiento — el ejemplo de arriba repite variantes de palabras clave
            en bucle —, aunque también hay fichas largas con descripciones legítimas:
            la cola es heterogénea, y precisamente por eso no se puede «limpiar a
            mano» con una regla simple. Esta observación condiciona el notebook 01:
            *qué* se codifica importa tanto como *con qué modelo*.

            Las marcas son un metadato razonablemente poblado — 4,4 % vacío, más un
            puñado de fichas con el literal «Desconocido», que es un vacío disfrazado
            y se trata como tal en el análisis — y por eso buen candidato a filtro
            estructurado:
            """
        ),
        code(
            r"""
            informative_brands = catalog.loc[
                ~catalog["brand"].isin(["", "Desconocido"]), "brand"
            ]
            print(f"Fichas sin marca informativa: "
                  f"{len(catalog) - len(informative_brands)} "
                  f"({100 * (1 - len(informative_brands) / len(catalog)):.1f}%)")
            top_brands = (
                informative_brands.value_counts()
                .head(15)
                .rename_axis("brand")
                .reset_index(name="productos")
            )
            px.bar(
                top_brands, x="brand", y="productos",
                title="Las 15 marcas (informativas) con más productos en el catálogo",
            )
            """
        ),
        markdown(
            r"""
            ## Las consultas y sus juicios: relevancia graduada

            Para medir un buscador hace falta saber qué respuestas son buenas. Los
            **juicios de relevancia** (*qrels*) son anotaciones humanas que etiquetan
            pares consulta-producto. El dataset ESCI usa cuatro grados en lugar de un
            binario relevante/irrelevante:

            | Etiqueta | Significado | Ganancia |
            |---|---|---|
            | **E** (Exact) | el producto es exactamente lo pedido | 3 |
            | **S** (Substitute) | sirve como sustituto razonable | 2 |
            | **C** (Complement) | complementa lo pedido (funda para el móvil buscado) | 1 |
            | **I** (Irrelevant) | no responde a la intención | 0 |

            Ocho consultas de desarrollo con 248 juicios graduados. La distribución
            de relevantes por consulta explica de antemano los techos de recall con
            k=10: hay consultas con hasta 39 relevantes.
            """
        ),
        code(
            r"""
            from aurum_discovery import load_development_judgments, load_development_queries

            queries = load_development_queries()
            judgments = load_development_judgments()
            per_query = (
                judgments.groupby(["query_id", "esci_label"]).size().unstack(fill_value=0)
            )
            per_query = per_query.join(
                queries.set_index("query_id")["query_text"].str.slice(0, 40)
            )
            per_query
            """
        ),
        markdown(
            r"""
            ## El baseline: BM25

            Antes de un solo embedding, una referencia interpretable. **BM25** es el
            estándar de la búsqueda léxica: puntúa un documento sumando, para cada
            término de la consulta que contiene, un peso que combina tres ideas:

            $$\text{score}(q, d) = \sum_{t \in q} \underbrace{\log\frac{N - n_t + 0.5}{n_t + 0.5}}_{\text{IDF: rareza del término}} \cdot \frac{f_{t,d}\,(k_1 + 1)}{f_{t,d} + k_1\left(1 - b + b\,\frac{|d|}{\overline{|d|}}\right)}$$

            - **IDF**: un término raro en el corpus (aparece en pocos de los $N$
              documentos) discrimina más que uno omnipresente.
            - **Saturación** ($k_1{=}1.5$): la quinta repetición de «taladro» aporta
              menos que la primera — la frecuencia $f_{t,d}$ satura.
            - **Normalización por longitud** ($b{=}0.75$): un documento largo no gana
              solo por tener más palabras.

            La tokenización pasa todo a minúsculas y elimina tildes («Tacón» y
            «tacon» deben coincidir). BM25 no entiende sinónimos ni intención: ese es
            precisamente su valor como baseline — todo lo que la representación densa
            mejore sobre él es semántica ganada, y donde no mejore, la semántica no
            aportaba.

            ## Las métricas, definidas antes de usarse

            Las tres métricas del proyecto se calculan sobre el top-10 devuelto
            ($k{=}10$) y se promedian entre consultas (**macro-media**: cada consulta
            pesa lo mismo, tenga 16 o 40 juicios):

            - **nDCG@k** (*normalized Discounted Cumulative Gain*) mide la calidad
              del **orden** con ganancias graduadas: cada resultado aporta su
              ganancia dividida por un descuento logarítmico de su posición,
              $DCG = \sum_{i=1}^{k} g_i / \log_2(i{+}1)$, y se normaliza por el DCG
              del orden ideal ($nDCG = DCG / IDCG \in [0, 1]$). Premia poner lo
              mejor arriba, no solo encontrarlo.
            - **Recall@k** mide la **cobertura**: fracción del conjunto relevante
              recuperada en el top-k. Aquí se declara relevante **E∪S** y el
              denominador es el conjunto relevante completo — con 39 relevantes y
              k=10 el techo es 10/39, y se reporta así, sin maquillar.
            - **MRR@k** (*Mean Reciprocal Rank*) mide la **primera satisfacción**:
              $1/\text{posición}$ del primer resultado relevante (aquí, solo **E**).
              Un MRR de 1.0 significa que el primer resultado ya es exacto.

            > ¡Ojo! Un producto recuperado **sin juicio** computa ganancia 0 en las
            > tres métricas, aunque a ojo parezca bueno. Con solo 248 juicios sobre
            > un catálogo de 15.000, las métricas absolutas subestiman al sistema;
            > por eso lo que se compara son **configuraciones entre sí**, siempre con
            > los mismos juicios. Esta declaración de relevancia queda fijada aquí y
            > no cambia en toda la serie.
            """
        ),
        code(
            r"""
            from time import perf_counter

            from aurum_discovery import Bm25Index, compose_document_text

            build_started = perf_counter()
            corpus = [
                compose_document_text(row, composition="full_text")
                for _, row in catalog.iterrows()
            ]
            bm25 = Bm25Index(corpus)
            print(f"Índice BM25 sobre {bm25.document_count} documentos en "
                  f"{perf_counter() - build_started:.1f}s")
            """
        ),
        code(
            r"""
            from aurum_discovery import evaluate_query

            product_ids = catalog["product_id"].tolist()
            baseline_rows = []
            for _, query in queries.iterrows():
                ranking = [
                    product_ids[position]
                    for position, _ in bm25.search(query["query_text"], top_k=10)
                ]
                metrics = evaluate_query(
                    query["query_id"], ranking, judgments, k=10,
                    recall_relevant_labels=run_config.recall_relevant_labels,
                    mrr_relevant_labels=run_config.mrr_relevant_labels,
                )
                baseline_rows.append(
                    {"query": query["query_text"][:38], **metrics.as_record()}
                )
            baseline_table = pd.DataFrame(baseline_rows).round(3)
            baseline_table
            """
        ),
        code(
            r"""
            macro = baseline_table[["ndcg_at_10", "recall_at_10", "mrr_at_10"]].mean()
            print("Macro-medias del baseline BM25:")
            print(macro.round(3).to_string())
            """
        ),
        markdown(
            r"""
            **Lectura.** BM25 es un rival serio en este catálogo — y conviene ser
            preciso sobre por qué: su corpus es el campo `text` **completo**, así que
            se beneficia tanto del vocabulario limpio de los títulos como del propio
            keyword-stuffing medido arriba, que para un buscador léxico funciona como
            recall gratuito (más palabras, más coincidencias). Ese es exactamente el
            listón correcto: la representación densa del notebook 01 solo se
            justifica si aporta sobre esto, y donde no aporte hay que decirlo.

            → Continúa en `actividad_01_representacion.ipynb`.
            """
        ),
    ]
    return cells
