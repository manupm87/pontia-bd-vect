"""Build the demo notebook cell by cell, mirroring the course sessions."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat import NotebookNode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "actividad_aurum_market.ipynb"

KERNEL_NAME = "aurum-market-eval"
KERNEL_DISPLAY_NAME = "Python (Aurum Market · Actividad)"


def markdown(source: str) -> NotebookNode:
    return nbformat.v4.new_markdown_cell(dedent(source).strip() + "\n")


def code(source: str) -> NotebookNode:
    return nbformat.v4.new_code_cell(dedent(source).strip() + "\n")


def build_cells() -> list[NotebookNode]:
    cells: list[NotebookNode] = []

    cells.extend(
        [
            markdown(
                r"""
                # Aurum Market · Demo de la entrega

                Este notebook recorre, de principio a fin, las decisiones que sostienen el
                motor de descubrimiento y control de catálogo: qué datos hay, qué
                representación ganó y por qué, cómo está configurada la base vectorial,
                cómo se busca (con y sin filtro), cuánto pierde el índice, cómo decide la
                regla de duplicados y qué contienen los artefactos entregados.

                **Requisitos previos** (una sola vez, desde el root del repo):

                ```bash
                make up && make embeddings && make pipeline
                ```

                Con eso Qdrant está arriba, la colección ingerida y los artefactos
                generados; cada celda de este notebook se apoya en ese estado y puede
                re-ejecutarse cuantas veces se quiera.

                ## Índice de contenidos

                1. [Los datos](#1.-Los-datos)
                2. [La representación: el experimento que decidió](#2.-La-representación:-el-experimento-que-decidió)
                3. [Índice y base de datos](#3.-Índice-y-base-de-datos)
                4. [Búsqueda semántica y filtros](#4.-Búsqueda-semántica-y-filtros)
                5. [Fidelidad ANN y latencia](#5.-Fidelidad-ANN-y-latencia)
                6. [Control de altas duplicadas](#6.-Control-de-altas-duplicadas)
                7. [Mutaciones y visibilidad](#7.-Mutaciones-y-visibilidad)
                8. [Los artefactos entregados](#8.-Los-artefactos-entregados)
                """
            ),
            code(
                r"""
                import sys
                from pathlib import Path

                project_root = Path.cwd().resolve()
                while not (project_root / "pyproject.toml").exists():
                    project_root = project_root.parent
                sys.path.insert(0, str(project_root / "src"))
                print(f"Root del proyecto: {project_root}")
                """
            ),
            code(
                r"""
                import json
                import os

                os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
                os.environ.setdefault("TQDM_DISABLE", "1")

                import pandas as pd
                import plotly.io as pio
                from dotenv import load_dotenv

                from aurum_discovery import (
                    load_catalog,
                    load_development_judgments,
                    load_development_queries,
                    load_manifest,
                    load_run_config,
                )

                load_dotenv(project_root / ".env")
                pio.templates.default = "plotly_white"
                pd.set_option("display.max_colwidth", 90)
                run_config = load_run_config()
                print(f"Configuración final: {run_config.embedding_configuration}")
                """
            ),
        ]
    )

    cells.extend(
        [
            markdown(
                r"""
                ## 1. Los datos

                El catálogo tiene 15.000 fichas en español con la suciedad típica de un
                marketplace real: metadatos ausentes, marcas escritas de formas distintas
                y descripciones con calidad desigual. El manifiesto conserva la
                procedencia (ESCI, Apache-2.0), el contrato de IDs y los checksums.
                """
            ),
            code(
                r"""
                catalog = load_catalog()
                manifest = load_manifest()
                print(f"Registros: {len(catalog)}")
                print(f"Snapshot: {manifest['snapshot_id']}")
                print(f"Marcas vacías: {(catalog['brand'] == '').sum()}")
                print(f"Colores vacíos: {(catalog['color'] == '').sum()}")
                catalog[["record_id", "product_id", "title", "brand"]].head(3)
                """
            ),
            markdown(
                r"""
                La longitud del campo `text` delata el problema que decidirá la sección 2:
                hay fichas cuyo "texto" es una ristra de miles de caracteres de palabras
                clave repetidas para posicionar el producto, no una descripción.
                """
            ),
            code(
                r"""
                text_lengths = catalog["text"].str.len()
                print(text_lengths.describe().round(0).to_string())
                worst = catalog.loc[text_lengths.idxmax()]
                print(f"\nEjemplo extremo ({text_lengths.max()} caracteres): {worst['title'][:80]}...")
                print(worst["text"][280:560])
                """
            ),
        ]
    )

    cells.extend(
        [
            markdown(
                r"""
                ## 2. La representación: el experimento que decidió

                Se compararon cuatro alternativas sobre las 8 consultas de desarrollo,
                siempre con **búsqueda exacta** para aislar la representación del índice:
                un baseline léxico BM25 y tres configuraciones densas (E5 con texto
                completo, E5 con título+marca+color, y MiniLM). La relevancia se declaró
                antes de medir y no cambió: nDCG con ganancias E=3/S=2/C=1/I=0, recall
                sobre E∪S, MRR sobre E.
                """
            ),
            code(
                r"""
                experiments_table = pd.read_csv(
                    project_root / ".artifacts" / "experimentos" / "tabla_comparativa.csv"
                )
                experiments_table.round(3)
                """
            ),
            markdown(
                r"""
                Dos lecturas: el campo `text` completo **perjudica** a E5 (el
                keyword-stuffing de la sección 1 contamina el embedding), y MiniLM se
                hunde porque carece del entrenamiento de recuperación con prefijos
                `query:`/`passage:` de E5. La celda siguiente lo hace tangible con una
                consulta de desarrollo: mismo modelo, distinta composición.
                """
            ),
            code(
                r"""
                import numpy as np

                from aurum_discovery import load_embedding_set

                def exact_top(configuration_name: str, workload_id: str, k: int = 3) -> pd.DataFrame:
                    embedding_set = load_embedding_set(configuration_name)
                    scores = embedding_set.matrix("products") @ embedding_set.vector(
                        "consultas_desarrollo", workload_id
                    )
                    positions = np.argsort(-scores)[:k]
                    rows = catalog.iloc[positions][["product_id", "title"]].copy()
                    rows.insert(0, "score", scores[positions].round(4))
                    rows["title"] = rows["title"].str.slice(0, 70)
                    return rows.assign(configuracion=configuration_name)

                queries = load_development_queries()
                demo_query = queries.iloc[1]
                print(f"Consulta: {demo_query['query_text']!r}")
                pd.concat(
                    [exact_top("e5_small_full", demo_query["workload_id"]),
                     exact_top("e5_small_title", demo_query["workload_id"])]
                ).set_index("configuracion")
                """
            ),
            markdown(
                r"""
                La configuración ganadora queda fijada en `config/run_config.yaml`
                (`e5_small_title`): es el contrato reproducible que consumen todos los
                scripts, de modo que ninguna métrica del informe depende de un estado
                oculto del notebook.
                """
            ),
        ]
    )

    cells.extend(
        [
            markdown(
                r"""
                ## 3. Índice y base de datos

                La colección usa distancia coseno, ID de punto igual al `record_id`
                UUIDv5 del catálogo, HNSW explícito (m=24, ef_construct=120) e índice de
                payload `keyword` sobre `brand`. Y una verificación aprendida por las
                malas: **el estado `green` no implica que el índice exista**. Qdrant solo
                construye el grafo HNSW cuando un segmento supera `indexing_threshold`;
                con el valor por defecto, 15.000 vectores quedaban en escaneo exhaustivo
                con `indexed_vectors_count=0`. Por eso la creación fija un umbral bajo y
                el arranque comprueba los vectores indexados, no el semáforo.
                """
            ),
            code(
                r"""
                import os

                from aurum_discovery import CatalogVectorStore, load_embedding_set

                embedding_set = load_embedding_set(run_config.embedding_configuration)
                store = CatalogVectorStore(
                    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                    api_key=os.getenv("QDRANT_API_KEY", ""),
                    collection_name=os.getenv("QDRANT_COLLECTION", "aurum-market-eval-catalogo"),
                    vector_size=embedding_set.configuration.dimension,
                    hnsw=run_config.hnsw,
                    ef_search=run_config.ef_search,
                )
                store.ping()
                index_state = store.wait_until_indexed()
                print(f"Registros: {store.count()}")
                print(f"Estado: {index_state}")
                assert index_state["indexed_vectors_count"] > 0, index_state
                """
            ),
            markdown(
                r"""
                La ingesta es **idempotente**: el ID de punto es el UUIDv5 estable del
                producto, así que repetir un lote hace *upsert*, no inserción. Reingerimos
                los primeros 256 registros y el recuento no se mueve.
                """
            ),
            code(
                r"""
                from aurum_discovery import iter_record_batches

                count_before = store.count()
                first_batch = next(
                    iter_record_batches(
                        catalog, embedding_set.matrix("products"), batch_size=256
                    )
                )
                store.upsert_records([first_batch])
                print(f"Antes: {count_before} · Después: {store.count()}")
                assert store.count() == count_before
                """
            ),
        ]
    )

    cells.extend(
        [
            markdown(
                r"""
                ## 4. Búsqueda semántica y filtros

                La interfaz común (`DiscoveryService`) recibe texto libre, aplica el
                prefijo `query:`, codifica y delega la búsqueda en la base de datos.
                El score devuelto conserva su semántica nativa: similitud coseno,
                mayor es mejor.
                """
            ),
            code(
                r"""
                from aurum_discovery import DiscoveryService, get_configuration

                service = DiscoveryService(
                    store=store,
                    configuration=get_configuration(run_config.embedding_configuration),
                )
                hits = service.search_text("taladro sin cable potente", top_k=5)
                pd.DataFrame(
                    [
                        {"rank": hit.rank, "score": round(hit.native_score, 4),
                         "brand": hit.brand, "title": hit.title[:70]}
                        for hit in hits
                    ]
                )
                """
            ),
            markdown(
                r"""
                El filtro de marca viaja **dentro** de la consulta (`query_filter` sobre
                el índice de payload), nunca como post-filtrado del top-10 global: la
                base devuelve el top-k *dentro* del universo filtrado.
                """
            ),
            code(
                r"""
                filtered_hits = service.search_text(
                    "herramienta inalámbrica para perforar", top_k=5, brand="Einhell"
                )
                brands = {hit.brand for hit in filtered_hits}
                assert brands == {"Einhell"}, brands
                pd.DataFrame(
                    [
                        {"rank": hit.rank, "score": round(hit.native_score, 4),
                         "brand": hit.brand, "title": hit.title[:70]}
                        for hit in filtered_hits
                    ]
                )
                """
            ),
            code(
                r"""
                no_results = service.search_text(
                    "herramienta inalámbrica", top_k=5, brand="MarcaInexistenteAurum"
                )
                print(f"Filtro sin resultados -> lista vacía sin error: {no_results}")
                """
            ),
        ]
    )

    cells.extend(
        [
            markdown(
                r"""
                ## 5. Fidelidad ANN y latencia

                La fidelidad compara los IDs del índice HNSW contra un **oráculo exacto
                en la misma colección** (`SearchParams.exact=true`), separando la pérdida
                del índice del error del modelo. Primero en vivo sobre las consultas de
                desarrollo:
                """
            ),
            code(
                r"""
                development_matrix = embedding_set.matrix("consultas_desarrollo")
                overlaps = []
                for row in range(development_matrix.shape[0]):
                    query_vector = development_matrix[row]
                    exact_ids = {hit.record_id for hit in store.search(query_vector, top_k=10, exact=True)}
                    ann_ids = {hit.record_id for hit in store.search(query_vector, top_k=10)}
                    overlaps.append(len(exact_ids & ann_ids) / 10)
                print(f"Fidelidad@10 por consulta: {overlaps}")
                print(f"Media: {sum(overlaps) / len(overlaps):.3f}")
                """
            ),
            markdown(
                r"""
                Que salga 1.0 no es una tautología: el barrido de `ef_search` generado en
                la evaluación demuestra que el índice **puede** perder (con `ef=10` la
                peor consulta cae al 20 % de fidelidad) y que a esta escala `ef=128`
                compra fidelidad perfecta sin coste de latencia apreciable.
                """
            ),
            code(
                r"""
                import plotly.express as px

                sweep = pd.DataFrame(
                    json.loads(
                        (project_root / ".artifacts" / "evaluacion" / "barrido_ef_search.json").read_text()
                    )["sweep"]
                )
                figure = px.line(
                    sweep, x="ef_search", y=["mean_overlap_at_10", "min_overlap_at_10"],
                    markers=True, title="Fidelidad ANN frente a ef_search (20 consultas)",
                    labels={"value": "fidelidad@10", "variable": "métrica"},
                )
                figure.update_layout(xaxis_title="ef_search")
                figure
                """
            ),
        ]
    )

    cells.extend(
        [
            markdown(
                r"""
                ## 6. Control de altas duplicadas

                La base vectorial genera los candidatos (la ficha entrante se codifica
                como documento) y una regla reproducible decide: duplicado si el mejor
                candidato alcanza `score >= 0.955`. El umbral salió de los 14 casos de
                desarrollo: los duplicados puntúan en [0.980, 1.000] y los no duplicados
                en [0.883, 0.931], y se eligió un valor centrado en ese hueco
                (*max-margin*) en vez de pegarlo al caso extremo.
                """
            ),
            code(
                r"""
                from aurum_discovery import DuplicateRule, load_incoming_products

                rule = DuplicateRule(
                    score_threshold=run_config.duplicate_score_threshold,
                    margin_threshold=run_config.duplicate_margin_threshold,
                )
                labeled = load_incoming_products(labeled=True).set_index("incoming_id")
                rows = []
                for incoming_id in ("DEV-DUP-006", "DEV-NEW-007"):
                    vector = embedding_set.vector("altas_desarrollo", incoming_id)
                    decision = rule.decide(incoming_id, store.search(vector, top_k=5))
                    rows.append(
                        {"incoming_id": incoming_id,
                         "etiqueta": labeled.loc[incoming_id, "is_duplicate"],
                         "prediccion": decision.predicted_duplicate,
                         "score": round(decision.score, 4),
                         "margen": round(decision.margin, 4),
                         "candidato": decision.matched_product_id}
                    )
                pd.DataFrame(rows)
                """
            ),
            markdown(
                r"""
                `DEV-NEW-007` es el caso frontera del dataset (un no-duplicado con score
                0.931, a 0.024 del umbral): el que habría que monitorizar en producción.
                La calibración completa —rejilla explorada, matriz de confusión y regla
                configurada— queda auditada en `.artifacts/duplicados/calibracion.json`.
                """
            ),
            code(
                r"""
                calibration = json.loads(
                    (project_root / ".artifacts" / "duplicados" / "calibracion.json").read_text()
                )
                configured_rule = calibration["configured_rule"]
                print(f"Umbral configurado: {configured_rule['rule']['score_threshold']}")
                print(
                    f"Desarrollo -> precision {configured_rule['precision']:.3f} · "
                    f"recall {configured_rule['recall']:.3f} · F1 {configured_rule['f1']:.3f}"
                )
                print(f"Rejilla explorada: {calibration['calibration']['explored']} combinaciones")
                """
            ),
        ]
    )

    cells.extend(
        [
            markdown(
                r"""
                ## 7. Mutaciones y visibilidad

                Los 24 eventos oficiales se aplican con `make events` sobre una colección
                dedicada (dos pasadas, recuento estable: la prueba de idempotencia está
                en el informe de eventos). Aquí demostramos el ciclo completo con un
                **registro canario**: alta, lectura por ID, actualización, búsqueda y
                eliminación, verificando la visibilidad tras cada paso. El borrado va en
                un `finally`, de modo que el canario desaparece del catálogo incluso si
                una verificación intermedia fallara: la celda es re-ejecutable sin dejar
                rastro.
                """
            ),
            code(
                r"""
                import numpy as np

                from aurum_discovery import VectorRecord, record_id_for_product

                canary_id = record_id_for_product("AURUM-DEMO-CANARY")
                canary_vector = np.zeros(embedding_set.configuration.dimension, dtype=np.float32)
                canary_vector[0] = 1.0

                def canary(version: int) -> VectorRecord:
                    return VectorRecord(
                        record_id=canary_id, product_id="AURUM-DEMO-CANARY",
                        title=f"Registro canario v{version}", brand="AurumDemo",
                        color="", locale="es", catalog_version=version, active=True,
                        text="Registro canario de demostración",
                        embedding=canary_vector.tolist(),
                    )
                """
            ),
            code(
                r"""
                try:
                    store.upsert_records([[canary(1)]])
                    assert store.retrieve(canary_id)["catalog_version"] == 1
                    store.upsert_records([[canary(2)]])
                    assert store.retrieve(canary_id)["catalog_version"] == 2
                    top = store.search(canary_vector, top_k=1)[0]
                    print(f"Visible en búsqueda: {top.title!r} (score {top.native_score:.3f})")
                finally:
                    store.delete_records([canary_id])

                assert store.retrieve(canary_id) is None
                survivors = {hit.record_id for hit in store.search(canary_vector, top_k=10)}
                assert canary_id not in survivors
                print(f"Canario eliminado; recuento estable: {store.count()}")
                """
            ),
            code(
                r"""
                events_report = json.loads(
                    (project_root / ".artifacts" / "eventos" / "informe_eventos.json").read_text()
                )
                print(f"Eventos aplicados: {events_report['event_count']} (dos pasadas)")
                print(
                    "Recuentos: "
                    f"{events_report['count_after_first_pass']} tras la primera, "
                    f"{events_report['count_after_second_pass']} tras la segunda "
                    f"(idempotente: {events_report['idempotent']})"
                )
                pd.DataFrame(events_report["visibility"])
                """
            ),
        ]
    )

    cells.extend(
        [
            markdown(
                r"""
                ## 8. Los artefactos entregados

                Todo lo anterior converge en los tres artefactos de `resultados/`,
                regenerables desde `config/run_config.yaml` con un comando cada uno.
                El top-10 ciego de una de las consultas de evaluación, con los títulos
                del catálogo para poder juzgarlo a ojo:
                """
            ),
            code(
                r"""
                search_results = pd.read_csv(project_root / "resultados" / "resultados_busqueda.csv")
                one_query = search_results[search_results["evaluation_id"] == "EVAL-100455-semantic"]
                one_query.merge(
                    catalog[["product_id", "title", "brand"]], on="product_id", how="left"
                ).assign(title=lambda frame: frame["title"].str.slice(0, 70))
                """
            ),
            code(
                r"""
                duplicate_results = pd.read_csv(
                    project_root / "resultados" / "resultados_duplicados.csv",
                    keep_default_na=False,
                )
                duplicate_results
                """
            ),
            code(
                r"""
                metrics = json.loads(
                    (project_root / "resultados" / "metricas_desarrollo.json").read_text()
                )
                {key: metrics[key] for key in (
                    "ndcg_at_10", "recall_at_10", "mrr_at_10",
                    "latency_p50_ms", "latency_p95_ms",
                    "ann_fidelity_mean_overlap_at_10", "record_count",
                )}
                """
            ),
            markdown(
                r"""
                ---

                **Cierre.** El sistema no se limita a devolver algo: mide cuándo sus
                resultados son útiles (nDCG/recall/MRR contra juicios graduados), sabe
                qué capa explica cada error (representación y juicios, no el índice:
                fidelidad 1.0 verificada contra oráculo exacto) y deja cada decisión
                —composición del texto, parámetros HNSW, umbral de duplicados— fijada en
                `config/run_config.yaml` y regenerable con `make pipeline`. El análisis
                completo está en `INFORME_AURUM_MARKET.pdf`.
                """
            ),
        ]
    )

    return cells


def main() -> None:
    """Assemble the notebook and write it with the session kernel metadata."""
    notebook = nbformat.v4.new_notebook(
        cells=build_cells(),
        metadata={
            "kernelspec": {
                "display_name": KERNEL_DISPLAY_NAME,
                "language": "python",
                "name": KERNEL_NAME,
            },
            "language_info": {"name": "python", "version": "3.12"},
            "case_study": "Aurum Market · Evaluación de Bases de Datos Vectoriales",
        },
    )
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"Notebook escrito en {NOTEBOOK_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
