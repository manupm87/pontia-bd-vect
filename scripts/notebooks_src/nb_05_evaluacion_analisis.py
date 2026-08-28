"""Notebook 05: full evaluation, error attribution, and the rubric checklist."""

from __future__ import annotations

from nbformat import NotebookNode

from .common import code, markdown, setup_cells, store_cells

FILENAME = "actividad_05_evaluacion_y_analisis.ipynb"


def build_cells() -> list[NotebookNode]:
    return [
        markdown(
            r"""
            # 05 · Evaluación y análisis

            **§5 del enunciado, evidencia a evidencia** — la evaluación sirve para
            decidir, no para decorar. Cada experimento conserva configuración,
            métricas e IDs recuperados, y todo se regenera con `make metrics`.

            | Evidencia | Mínimo exigido | Dónde se demuestra aquí |
            |---|---|---|
            | Calidad del ranking | nDCG@10, Recall@10, MRR@10 sobre desarrollo | §1 |
            | Fidelidad ANN | IDs frente a oráculo exacto sobre una muestra | §2 |
            | Latencia | p50 y p95 con entorno, calentamiento y repeticiones | §3 |
            | Filtros | resultados que cumplen la marca en las 4 consultas | §4 |
            | Duplicados | precision, recall y F1 con umbral documentado | §4 |
            | Mutaciones | recuento, lectura por ID, búsqueda y estado tras 24 eventos | §4 |
            | Atribución de errores | ≥3 fallos con capa responsable y evidencia | §5 |
            """
        ),
        *setup_cells(),
        *store_cells(),
        markdown(
            r"""
            ## 1. Calidad del ranking

            Métricas de la configuración final (`e5_small_title`) medidas **a través
            de la base de datos** (ruta de producción, índice HNSW incluido), con la
            declaración de relevancia fijada desde el notebook 00: nDCG graduado
            (E=3/S=2/C=1/I=0), recall sobre E∪S con el conjunto relevante completo
            como denominador, MRR sobre E.
            """
        ),
        code(
            r"""
            metrics = json.loads(
                (project_root / "resultados" / "metricas_desarrollo.json").read_text()
            )
            print(f"Relevantes para recall: {metrics['recall_relevant_labels']} · "
                  f"para MRR: {metrics['mrr_relevant_labels']}")
            pd.DataFrame(
                [{k: metrics[k] for k in ("ndcg_at_10", "recall_at_10", "mrr_at_10")}]
            )
            """
        ),
        code(
            r"""
            evaluation_run = json.loads(
                (project_root / ".artifacts" / "evaluacion" / "evaluation_run.json").read_text()
            )
            per_query = pd.DataFrame(evaluation_run["per_query"]).round(3)
            per_query
            """
        ),
        code(
            r"""
            import plotly.express as px

            long_metrics = per_query.melt(
                id_vars=["query_id", "judged_count", "relevant_count"],
                value_vars=["ndcg_at_10", "recall_at_10", "mrr_at_10"],
                var_name="métrica", value_name="valor",
            )
            px.bar(
                long_metrics, x="query_id", y="valor", color="métrica", barmode="group",
                title="Métricas por consulta de desarrollo (configuración final, vía Qdrant)",
            )
            """
        ),
        markdown(
            r"""
            La dispersión es la historia real: cinco consultas con MRR=1.0 conviven
            con dos fallos claros (33633 y 38249) que la sección de atribución
            disecciona. El recall bajo en términos absolutos tiene techo estructural:
            con hasta 39 relevantes y k=10, el máximo alcanzable es 10/39 — se reporta
            con su denominador honesto en vez de maquillarlo.

            ## 2. Fidelidad ANN: el índice, contra su oráculo

            El índice HNSW es *aproximado* (notebook 02): puede perder vecinos
            verdaderos. La **fidelidad** cuantifica cuánto, comparando IDs — nunca
            scores — contra el oráculo exacto:

            $$\text{fidelidad@k} = \frac{|\,\text{top-k exacto} \cap \text{top-k ANN}\,|}{k}$$

            Es una métrica *algorítmica*, no de negocio: no dice si los resultados
            son buenos (eso es el nDCG de §1, contra juicios humanos), dice si el
            índice devuelve **lo mismo que devolvería la fuerza bruta**. Mantener
            ambas separadas es lo que luego permite atribuir errores: un fallo con
            fidelidad 1.0 no puede ser culpa del índice. El oráculo es la **misma
            colección en modo exhaustivo** (`SearchParams.exact=true`), sobre las 20
            consultas (8 desarrollo + 12 evaluación) — medido en vivo ahora mismo:
            """
        ),
        code(
            r"""
            import numpy as np

            fidelity_vectors = np.concatenate(
                [embedding_set.matrix("consultas_desarrollo"),
                 embedding_set.matrix("consultas_evaluacion")]
            )
            overlaps = []
            for row in range(fidelity_vectors.shape[0]):
                query_vector = fidelity_vectors[row]
                exact_ids = {hit.record_id for hit in store.search(query_vector, top_k=10, exact=True)}
                ann_ids = {hit.record_id for hit in store.search(query_vector, top_k=10)}
                overlaps.append(len(exact_ids & ann_ids) / 10)
            print(f"Fidelidad@10 con ef_search={run_config.ef_search}: "
                  f"media {np.mean(overlaps):.3f} · mínima {np.min(overlaps):.3f} "
                  f"({len(overlaps)} consultas)")
            """
        ),
        markdown(
            r"""
            Una fidelidad de 1.0 solo es creíble si se demuestra que la medición
            **puede** detectar pérdida. El barrido de `ef_search` (`make sweep-ef` →
            `.artifacts/evaluacion/barrido_ef_search.json`) mide dos grafos sobre el
            mismo catálogo: la configuración entregada (m=24, `ef_construct=120`,
            totalmente optimizada) y un **grafo deliberadamente débil** (m=4,
            `ef_construct=16`) en una colección dedicada, donde el compromiso
            fidelidad/esfuerzo emerge con claridad.
            """
        ),
        code(
            r"""
            sweep_report = json.loads(
                (project_root / ".artifacts" / "evaluacion" / "barrido_ef_search.json").read_text()
            )
            frames = []
            for graph_name, block in (("m=24 (entregado)", sweep_report["final_config"]),
                                      ("m=4 (débil)", sweep_report["weak_graph"])):
                frame = pd.DataFrame(block["sweep"])
                frame["grafo"] = graph_name
                frames.append(frame)
            sweep = pd.concat(frames)
            figure = px.line(
                sweep, x="ef_search", y="mean_overlap_at_10", color="grafo",
                markers=True, labels={"mean_overlap_at_10": "fidelidad media@10"},
                title="Fidelidad frente a ef_search: el grafo entregado y uno débil (20 consultas)",
            )
            figure.update_yaxes(range=[0, 1.05])
            figure
            """
        ),
        code(
            r"""
            sweep[["grafo", "ef_search", "mean_overlap_at_10", "min_overlap_at_10", "p50_ms"]]
            """
        ),
        markdown(
            r"""
            Dos lecciones, una por grafo:

            1. **El grafo entregado no pierde nada**: con m=24 y el índice
               plenamente optimizado, la fidelidad es 1.0 en todo el barrido —
               incluso a `ef=10`. A 15.000 vectores, un grafo bien construido hace
               que `ef_search` sea un margen de seguridad, no un dial crítico. (No
               siempre fue así: durante la construcción, con los segmentos a medio
               fusionar, este mismo barrido llegó a medir pérdidas del 80 % en una
               consulta a `ef=10` — la fidelidad depende del *estado* del grafo,
               otra razón para medirla en lugar de suponerla.)
            2. **Un grafo mal construido no se arregla buscando más**: con m=4 la
               fidelidad media cae a 0.645 a `ef=10` (mínima 0.20) y, aunque sube
               con `ef`, se estanca en ~0.87 a `ef=128`: las aristas que no existen
               no se pueden recorrer. La calidad se decide en `m`/`ef_construct` (en
               construcción), y `ef_search` solo explota el grafo que hay.

            Al crecer el catálogo, este barrido es la herramienta de re-calibración
            (y la separación entre «pérdida del índice» y «error del modelo» que
            exige la atribución de errores).

            ## 3. Latencia: protocolo declarado, no ranking de nubes

            La latencia no se resume con una media: unas pocas consultas lentas la
            distorsionan y esconden justo lo que interesa. Se usan **percentiles**:
            el **p50** (la mediana — la mitad de las consultas responde más rápido)
            describe la experiencia típica, y el **p95** describe la cola — el peor
            caso que aún vive el 5 % de las peticiones, que en un servicio real es lo
            que dispara los timeouts. Dos disciplinas más del protocolo: **rondas de
            calentamiento** previas y descartadas (las primeras peticiones pagan
            cachés frías y conexiones nuevas que no representan el régimen
            estacionario) y **repeticiones** suficientes para que los percentiles se
            estabilicen.

            Aquí: p50/p95 sobre las 12 consultas de evaluación, 5 rondas de
            calentamiento y 30 repeticiones, medido ahora mismo por la ruta de
            producción. El entorno queda registrado junto a la métrica; el enunciado
            prohíbe usar esto para comparar proveedores en infraestructuras
            distintas, y no se hace.
            """
        ),
        code(
            r"""
            from aurum_discovery import measure_latency

            evaluation_matrix = embedding_set.matrix("consultas_evaluacion")
            operations = [
                (lambda vector=evaluation_matrix[row]: store.search(vector, top_k=10))
                for row in range(evaluation_matrix.shape[0])
            ]
            latency = measure_latency(
                operations, warmup=run_config.latency_warmup,
                repeats=run_config.latency_repeats,
            )
            print(f"En vivo -> p50 {latency.p50_ms:.2f} ms · p95 {latency.p95_ms:.2f} ms "
                  f"(warmup={latency.warmup}, repeats={latency.repeats})")
            print(f"Entregado -> p50 {metrics['latency_p50_ms']} ms · "
                  f"p95 {metrics['latency_p95_ms']} ms (varía ±0.3 ms entre ejecuciones)")
            print(f"Entorno: {evaluation_run['environment']['platform']}")
            """
        ),
        markdown(
            r"""
            ## 4. Filtros, duplicados y mutaciones (evidencias cerradas)

            Verificadas en vivo en sus notebooks (03 y 04); aquí, el resumen con sus
            artefactos fuente:
            """
        ),
        code(
            r"""
            filters_report = json.loads(
                (project_root / ".artifacts" / "filtros" / "informe_filtros.json").read_text()
            )
            events_report = json.loads(
                (project_root / ".artifacts" / "eventos" / "informe_eventos.json").read_text()
            )
            duplicates_block = metrics["duplicados_desarrollo"]
            summary = pd.DataFrame(
                [
                    {"evidencia": "filtros",
                     "resultado": f"{len(filters_report['queries'])}/4 consultas, todas cumplen la marca; "
                     f"marca inexistente -> {filters_report['empty_filter_probe']['result_count']} resultados"},
                    {"evidencia": "duplicados (desarrollo)",
                     "resultado": f"precision {duplicates_block['precision']:.2f} · recall {duplicates_block['recall']:.2f} · "
                     f"F1 {duplicates_block['f1']:.2f} con umbral {duplicates_block['score_threshold']}"},
                    {"evidencia": "mutaciones",
                     "resultado": f"{events_report['event_count']} eventos x2 pasadas -> recuento "
                     f"{events_report['count_after_second_pass']} (idempotente: {events_report['idempotent']})"},
                ]
            )
            summary
            """
        ),
        markdown(
            r"""
            ## 5. Atribución de errores: los fallos y sus capas

            Cuando un ranking falla, la pregunta útil no es «¿falló?» sino «¿**qué
            capa** falló?» — porque cada capa se arregla distinto. El método de
            diagnóstico, con su prueba discriminante:

            | Capa | El error es suyo si… | Cómo se comprueba |
            |---|---|---|
            | Representación | el vecino **exacto** ya es semánticamente malo | el oráculo devuelve lo mismo que el ANN |
            | Índice | el oráculo recupera algo que el ANN pierde | fidelidad@k < 1 en esa consulta |
            | Datos / filtros | falta información, el metadato es inconsistente o el juicio es ruidoso | inspección de qrels y payload |
            | Persistencia / consistencia | lo leído no refleja aún lo escrito | verificaciones de visibilidad |

            Los tres fallos reales del sistema caen en dos capas (representación ×2,
            datos/juicios ×1); las otras dos capas se cierran con su propia
            evidencia — la de índice **provocándole una pérdida real** al final de
            esta sección, y la de persistencia con las verificaciones de visibilidad
            del notebook 04. Para cada fallo se muestra la evidencia, no solo el
            veredicto, empezando por el descarte del índice **en la propia consulta
            analizada** (no solo en la media agregada):
            """
        ),
        code(
            r"""
            from aurum_discovery import load_catalog, load_development_judgments

            catalog = load_catalog().set_index("product_id")
            judgments = load_development_judgments()
            rankings = evaluation_run["development_rankings"]

            def attribution_table(workload_id: str, query_id: str) -> pd.DataFrame:
                judged = judgments[judgments["query_id"] == query_id].set_index("product_id")
                rows = []
                for rank, product_id in enumerate(rankings[workload_id], start=1):
                    label = judged.loc[product_id, "esci_label"] if product_id in judged.index else "sin juicio"
                    rows.append(
                        {"rank": rank, "etiqueta": label,
                         "brand": catalog.loc[product_id, "brand"] if product_id in catalog.index else "?",
                         "title": str(catalog.loc[product_id, "title"])[:64] if product_id in catalog.index else "?"}
                    )
                return pd.DataFrame(rows)
            """
        ),
        markdown(
            r"""
            ### Caso 1 · Representación — «estantes sin taladro habitacion» (38249, MRR 0.25)

            El top-3 son un libro de cocina *sin sal*, un disco y una novela: E5 ancla
            el patrón «sin + sustantivo» en títulos cortos sin marca. El oráculo exacto
            devuelve lo mismo (fidelidad 1.0), luego **el vecino exacto ya es malo: la
            capa responsable es la representación**. Mitigación razonable: señal de
            categoría en el texto o un reranking léxico ligero del top-50.
            """
        ),
        code(
            r"""
            def index_ruled_out(workload_id: str) -> None:
                vector = embedding_set.vector("consultas_desarrollo", workload_id)
                exact_ids = {hit.record_id for hit in store.search(vector, top_k=10, exact=True)}
                ann_ids = {hit.record_id for hit in store.search(vector, top_k=10)}
                assert exact_ids == ann_ids, (workload_id, exact_ids ^ ann_ids)
                print(f"{workload_id}: el top-10 del HNSW es idéntico al del oráculo "
                      "exacto — el índice queda descartado en esta consulta.")

            index_ruled_out("DEV-38249")
            attribution_table("DEV-38249", "38249")
            """
        ),
        markdown(
            r"""
            ### Caso 2 · Datos y juicios — «disfraz halloween talla grande hombre» (33633, nDCG 0.088)

            El sistema devuelve disfraces de Halloween reales, pero esta consulta
            tiene solo 16 juicios (de los cuales apenas 4 son relevantes E∪S) y el
            único *Exact* etiquetado es una blusa (ruido del dataset): los
            recuperados sin juicio computan ganancia 0.
            Hay un matiz de representación (ignora «talla grande hombre»), pero la
            magnitud del 0.088 la explica **la capa de datos/etiquetas** — ninguna
            configuración puede remontar juicios escasos y ruidosos.
            """
        ),
        code(
            r"""
            judged_33633 = judgments[judgments["query_id"] == "33633"]
            exact_labeled = judged_33633[judged_33633["esci_label"] == "E"]["product_id"].tolist()
            print(f"Juicios disponibles: {len(judged_33633)} · etiquetados E: {exact_labeled}")
            print(f"Título del único E: {catalog.loc[exact_labeled[0], 'title'][:80]}")
            attribution_table("DEV-33633", "33633").head(6)
            """
        ),
        markdown(
            r"""
            ### Caso 3 · Representación en atributos finos — «botines marrones mujer tacon medio» (18868, nDCG 0.329)

            El puesto 2 es un botín *plateado de tacón bajo* etiquetado **I**: la
            categoría se acierta, pero color y altura de tacón se difuminan en la
            similitud coseno de un modelo pequeño. La mitigación estructural no es
            otro modelo, sino mover los atributos duros (color) a metadatos
            filtrables, como ya se hace con la marca.
            """
        ),
        code(
            r"""
            index_ruled_out("DEV-18868")
            attribution_table("DEV-18868", "18868").head(6)
            """
        ),
        markdown(
            r"""
            ### La capa de índice, demostrada con una pérdida real

            Que ningún fallo del sistema sea del índice no significa que el índice
            no pueda fallar: la forma de evidencia que pide el enunciado — «el
            oráculo exacto recupera un elemento que el ANN pierde» — se materializa
            en el grafo débil (m=4) del barrido de §2. Localizamos en vivo una
            consulta donde ocurre y mostramos exactamente qué productos pierde:
            """
        ),
        code(
            r"""
            from aurum_discovery import HnswSettings

            weak_block = sweep_report["weak_graph"]
            weak_store = CatalogVectorStore(
                url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                collection_name=weak_block["collection"],
                vector_size=embedding_set.configuration.dimension,
                hnsw=HnswSettings(**weak_block["hnsw"]),
                ef_search=10,
            )
            all_ids = (embedding_set.identifiers["consultas_desarrollo"]
                       + embedding_set.identifiers["consultas_evaluacion"])
            worst_query = weak_block["sweep"][0]["worst_query_id"]
            position = all_ids.index(worst_query)
            exact_hits = store.search(fidelity_vectors[position], top_k=10, exact=True)
            weak_ids = {hit.record_id for hit in weak_store.search(fidelity_vectors[position], top_k=10)}
            lost = [hit for hit in exact_hits if hit.record_id not in weak_ids]
            print(f"Consulta {worst_query}, grafo m=4 con ef_search=10: el ANN pierde "
                  f"{len(lost)} de los 10 vecinos exactos, por ejemplo:")
            for hit in lost[:3]:
                print(f"  - (rank exacto {hit.rank}) {hit.title[:70]}")
            """
        ),
        markdown(
            r"""
            Esto cierra el método: cuando la fidelidad baja de 1.0, los productos
            perdidos son atribuibles al índice — y el remedio depende del origen
            (subir `ef_search` si el grafo es bueno; reconstruir con mejores
            `m`/`ef_construct` si no lo es). Cuando es 1.0, como en la configuración
            entregada, el índice queda fuera de toda sospecha y los fallos han de
            buscarse en la representación o en los datos.

            **Persistencia/consistencia**: cuarta capa del enunciado, sin fallos
            observados — las escrituras usan `wait=True`, las tres verificaciones de
            visibilidad del notebook 04 convergieron al primer intento y el estado
            final de los 24 eventos se validó registro a registro.

            ## 6. Antes de entregar: el checklist del enunciado, verificado en vivo
            """
        ),
        code(
            r"""
            blind = pd.read_csv(project_root / "resultados" / "resultados_busqueda.csv")
            blind_groups = blind.groupby("evaluation_id")
            assert len(blind_groups) == 12
            assert (blind_groups["product_id"].nunique() == 10).all()
            assert (blind_groups["rank"].min() == 1).all() and (blind_groups["rank"].max() == 10).all()
            catalog_products = set(catalog.index)
            assert set(blind["product_id"]) <= catalog_products
            print("✔ Los 12 rankings ciegos tienen 10 IDs únicos, válidos y rank 1..10")

            duplicates = pd.read_csv(
                project_root / "resultados" / "resultados_duplicados.csv",
                dtype=str, keep_default_na=False,
            )
            assert set(duplicates["predicted_duplicate"]) <= {"true", "false"}
            positives = duplicates[duplicates["predicted_duplicate"] == "true"]
            negatives = duplicates[duplicates["predicted_duplicate"] == "false"]
            assert len(positives) + len(negatives) == 14 and len(positives) > 0
            assert (positives["matched_product_id"] != "").all()
            assert (negatives["matched_product_id"] == "").all()
            assert set(positives["matched_product_id"]) <= catalog_products
            print(f"✔ Las {len(positives)} predicciones positivas señalan candidato válido "
                  f"y las {len(negatives)} negativas van sin candidato")
            """
        ),
        code(
            r"""
            assert store.count() == 15_000
            print("✔ La ingesta repetida no aumentó el recuento (15.000)")
            for query in filters_report["queries"]:
                assert query["all_results_match_brand"], query["workload_id"]
            print("✔ Las consultas filtradas nunca devuelven otra marca")
            assert events_report["idempotent"]
            print("✔ Los eventos dejan exactamente el estado esperado (dos pasadas)")
            for key in ("ndcg_at_10", "recall_at_10", "mrr_at_10", "latency_p50_ms", "latency_p95_ms"):
                assert key in metrics, key
            print("✔ metricas_desarrollo.json contiene las claves mínimas (make metrics)")
            gitignore = (project_root / ".gitignore").read_text(encoding="utf-8")
            assert ".env" in gitignore.splitlines()
            env_template = (project_root / ".env.example").read_text(encoding="utf-8")
            assert "HF_TOKEN=\n" in env_template and "QDRANT_API_KEY=\n" in env_template
            print("✔ Sin claves en el repo: .env ignorado por git, plantilla sin secretos")
            """
        ),
        markdown(
            r"""
            ## 7. Mapa contra los criterios de evaluación

            | Bloque (peso) | Dónde está la evidencia |
            |---|---|
            | Problema y baseline (10 %) | Notebook 00: contrato, datos, BM25 con las mismas métricas |
            | Representación vectorial (20 %) | Notebook 01: composición, prefijos, normalización, 4 experimentos con tablas y gráficos |
            | Índice y base de datos (25 %) | Notebook 02: esquema, HNSW explícito, `indexing_threshold`, ingesta idempotente, persistencia · Notebook 04 §4.1: mutaciones |
            | Recuperación y evaluación (30 %) | Notebook 03: interfaz y filtros · Notebook 04 §4.2: duplicados · este notebook: métricas, fidelidad, atribución |
            | Ingeniería y comunicación (15 %) | README, Makefile (`make pipeline`, `make metrics`), 55 tests, seguridad operativa, `INFORME_AURUM_MARKET.pdf`, `docs/images/arquitectura.svg` |

            Los artefactos entregables viven en `resultados/`; la configuración exacta
            de la ejecución final, en `config/run_config.yaml`. Fin de la serie.
            """
        ),
    ]
