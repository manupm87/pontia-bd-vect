"""Notebook 01: text composition, model choice, and the deciding experiments."""

from __future__ import annotations

from nbformat import NotebookNode

from .common import code, markdown, setup_cells

FILENAME = "actividad_01_representacion.ipynb"


def build_cells() -> list[NotebookNode]:
    return [
        markdown(
            r"""
            # 01 · Representación vectorial

            Este notebook justifica el **bloque 2** de los criterios: qué texto se
            codifica, con qué modelo, con qué prefijos y normalización — y la evidencia
            experimental que decidió cada elección. Regla del enunciado respetada al
            pie de la letra: *cambiar solo el nombre del modelo sin analizar el
            resultado no constituye un experimento*.

            ## Qué es exactamente un embedding (y por qué el coseno)

            Un **modelo de embeddings** es una red neuronal entrenada para que
            textos con significado parecido acaben en vectores cercanos. Los dos
            modelos de este notebook producen vectores de **384 dimensiones**; la
            diferencia está en *para qué* fueron entrenados, no en su tamaño:

            - **multilingual-e5-small** se entrenó con pares consulta-documento
              (objetivo de *recuperación*): aprende a acercar una pregunta a su
              respuesta, aunque no compartan vocabulario.
            - **paraphrase-multilingual-MiniLM** se entrenó con pares de paráfrasis:
              aprende a acercar frases que *dicen lo mismo*, que no es lo mismo que
              acercar una consulta a un producto.

            La cercanía se mide con la **similitud coseno**: el coseno del ángulo
            entre dos vectores,

            $$\cos(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\lVert\mathbf{q}\rVert \, \lVert\mathbf{d}\rVert} \in [-1, 1]$$

            que compara *dirección* (significado) ignorando *longitud* (cuánto texto
            había). Si los vectores se **L2-normalizan** (se reescalan a norma 1) al
            codificar, el denominador vale 1 y el coseno se reduce a un producto
            escalar — más barato de computar y con una consecuencia práctica: el
            score que devuelva la base de datos será directamente esta similitud,
            con una única semántica («mayor es mejor») en todo el sistema.

            Queda un detalle que decide experimentos enteros: los modelos E5 se
            entrenaron marcando cada texto con un **prefijo asimétrico** — `query:`
            para consultas y `passage:` para documentos. El modelo aprendió
            geometrías distintas para cada rol (una consulta de tres palabras y una
            ficha de trescientas no deben tratarse igual); omitir los prefijos lo
            saca de la distribución con la que fue entrenado y degrada el resultado.
            MiniLM no los usa: se entrenó sin roles.

            ## Las candidatas

            | Configuración | Modelo | Texto codificado | Prefijos |
            |---|---|---|---|
            | `e5_small_full` | multilingual-e5-small | campo `text` completo | `passage:` / `query:` |
            | `e5_small_title` | multilingual-e5-small | título + marca + color | `passage:` / `query:` |
            | `minilm_full` | paraphrase-multilingual-MiniLM | campo `text` completo | ninguno |
            | `bm25_full_text` | BM25 (léxico) | campo `text` completo | — |

            Saneado común: los valores vacíos son información ausente y **se omiten**
            (nunca se codifica la cadena `"nan"`); si `text` está vacío se recompone
            desde el título. Los embeddings se **L2-normalizan** al codificar, de modo
            que el producto escalar es la similitud coseno y el score conserva una
            semántica única en todo el sistema.
            """
        ),
        *setup_cells(),
        markdown(
            r"""
            ## La composición en la práctica

            La misma ficha, codificada de las dos formas. La versión `full_text`
            arrastra el keyword-stuffing medido en el notebook 00; la versión
            `title_brand_color` conserva la señal y elimina el ruido.
            """
        ),
        code(
            r"""
            from aurum_discovery import compose_document_text, load_catalog

            catalog = load_catalog()
            noisy_row = catalog.loc[catalog["text"].str.len().idxmax()]
            full_text = compose_document_text(noisy_row, composition="full_text")
            title_text = compose_document_text(noisy_row, composition="title_brand_color")
            print(f"full_text ({len(full_text)} caracteres):\n{full_text[:220]}...\n")
            print(f"title_brand_color ({len(title_text)} caracteres):\n{title_text}")
            """
        ),
        markdown(
            r"""
            ## El experimento que decidió

            Las cuatro alternativas se midieron sobre las 8 consultas de desarrollo
            (métricas definidas en el notebook 00) usando **búsqueda exacta**:
            calcular la similitud de la consulta contra los 15.000 productos y
            quedarse con el top-10 verdadero, sin ningún índice aproximado de por
            medio. Ese cálculo exhaustivo se llama **oráculo** y reaparecerá en los
            notebooks 02 y 05: aquí garantiza que lo que se compara es la
            *representación*, no el índice. El registro completo — configuración,
            métricas por consulta e IDs recuperados — está en
            `.artifacts/experimentos/registro_experimentos.json` y se regenera con
            `make experiments`.
            """
        ),
        code(
            r"""
            experiments = json.loads(
                (project_root / ".artifacts" / "experimentos" / "registro_experimentos.json").read_text()
            )
            macro_table = pd.DataFrame(
                [
                    {"experimento": exp["experiment"], **exp["metrics"]}
                    for exp in experiments["experiments"]
                ]
            ).round(3)
            macro_table
            """
        ),
        code(
            r"""
            import plotly.express as px

            long_macro = macro_table.melt(
                id_vars="experimento", var_name="métrica", value_name="valor"
            )
            px.bar(
                long_macro, x="métrica", y="valor", color="experimento",
                barmode="group", title="Macro-medias sobre las 8 consultas de desarrollo",
            )
            """
        ),
        markdown(
            r"""
            Tres lecturas, en orden de importancia:

            1. **La composición manda**: mismo modelo, y `title_brand_color` supera a
               `full_text` en las tres métricas. El texto sucio no es gratis.
            2. **El entrenamiento de recuperación manda más que la dimensión**: MiniLM
               (384d, como E5) se hunde sin prefijos `query:`/`passage:` ni objetivo de
               recuperación.
            3. **BM25 no desaparece**: gana por poco en nDCG y pierde en recall y MRR.
               La ventaja densa está en el emparejamiento por intención, no en el
               orden fino de lo ya encontrado léxicamente.
            """
        ),
        code(
            r"""
            per_query_frames = []
            for exp in experiments["experiments"]:
                frame = pd.DataFrame(exp["per_query"])
                frame["experimento"] = exp["experiment"]
                per_query_frames.append(frame)
            per_query = pd.concat(per_query_frames)
            px.bar(
                per_query, x="query_id", y="ndcg_at_10", color="experimento",
                barmode="group", title="nDCG@10 por consulta: dónde gana (y pierde) cada representación",
            )
            """
        ),
        markdown(
            r"""
            El desglose por consulta evita conclusiones de trazo grueso:
            `e5_small_title` no gana en todas partes (33633 es mala para todos — el
            notebook 05 la disecciona — y BM25 la supera en nDCG en 13357, 28703 y
            38249). Los derrumbes a cero solo los sufren `e5_small_full` (una
            consulta) y `minilm_full` (tres); BM25 tampoco se hunde, pero paga su
            dependencia léxica donde más duele en descubrimiento: en el **MRR**,
            `e5_small_title` pone un resultado exacto en primera posición en 5 de 8
            consultas frente a 4 de BM25, y en 13357 y 18868 la diferencia es 1.0
            frente a 0.167 y 0.5.
            """
        ),
        markdown(
            r"""
            ## Dónde aporta la semántica: consultas sin las palabras del título

            La prueba de fuego de una representación densa es la consulta que
            *describe* la intención sin usar el vocabulario del catálogo. Comparamos
            **el mismo BM25 de la tabla comparativa** (corpus `full_text`, k1=1.5,
            b=0.75 — sin rebajarle nada) contra la configuración ganadora en una
            formulación semántica del conjunto ciego.
            """
        ),
        code(
            r"""
            import numpy as np

            from aurum_discovery import Bm25Index, load_embedding_set, load_evaluation_queries

            evaluation_queries = load_evaluation_queries().set_index("evaluation_id")
            semantic_query = evaluation_queries.loc["EVAL-100455-semantic", "query_text"]
            print(f"Consulta: {semantic_query!r}\n")

            embedding_set = load_embedding_set(run_config.embedding_configuration)
            scores = embedding_set.matrix("products") @ embedding_set.vector(
                "consultas_evaluacion", "EVAL-100455-semantic"
            )
            dense_positions = np.argsort(-scores)[:5]
            dense_top = catalog.iloc[dense_positions][["title", "brand"]].copy()
            dense_top.insert(0, "score", scores[dense_positions].round(4))
            dense_top.assign(title=dense_top["title"].str.slice(0, 70))
            """
        ),
        code(
            r"""
            official_corpus = [
                compose_document_text(row, composition="full_text")
                for _, row in catalog.iterrows()
            ]
            bm25 = Bm25Index(official_corpus)
            lexical_top = [
                {"score": round(score, 2), "title": catalog.iloc[position]["title"][:70]}
                for position, score in bm25.search(semantic_query, top_k=5)
            ]
            pd.DataFrame(lexical_top)
            """
        ),
        markdown(
            r"""
            La diferencia no es caricaturesca, y por eso es creíble: el BM25 oficial
            también encuentra herramientas (el keyword-stuffing del campo `text` le
            regala recall), pero su podio lo encabezan una **aspiradora Dyson «sin
            cable»** y un router «inalámbrico» — coincidencias de palabras sueltas
            («sin», «inalámbrica», «potente») sin la intención — y el primer taladro
            aparece en 4ª posición. La representación densa pone cinco taladros a
            batería en las cinco primeras posiciones: entiende que la frase describe
            *perforar sin enchufe*. Es la misma diferencia que las métricas agregadas
            recogen como MRR (primera satisfacción): el usuario del buscador ve el
            primer resultado, no el cuarto.

            ## Decisión

            **`e5_small_title`** queda fijada en `config/run_config.yaml` como
            configuración de la ejecución final: multilingual-e5-small, título+marca+
            color, prefijos `query:`/`passage:`, L2-normalización y métrica coseno.
            El manifiesto de embeddings (`data/embeddings/e5_small_title/embedding_metadata.json`)
            encadena checksums SHA-256 de entradas y salidas para que el experimento
            sea auditable.

            → Continúa en `actividad_02_indice_y_bbdd.ipynb`.
            """
        ),
    ]
