"""Notebook 03: the common retrieval interface, brand filters, edge cases."""

from __future__ import annotations

from nbformat import NotebookNode

from .common import code, markdown, setup_cells, store_cells

FILENAME = "actividad_03_recuperacion_y_filtros.ipynb"


def build_cells() -> list[NotebookNode]:
    return [
        markdown(
            r"""
            # 03 · Recuperación y filtros

            **§3.3 del enunciado**: una interfaz común que recibe una consulta y
            devuelve resultados normalizados (`product_id`, posición, título,
            metadatos y score con semántica declarada), búsqueda global con top-k
            configurable, filtro de marca **ejecutado por la base de datos** y
            tratamiento explícito de los tres casos límite: colección vacía, filtro
            sin resultados y proveedor no disponible.
            """
        ),
        *setup_cells(),
        *store_cells(),
        markdown(
            r"""
            ## La interfaz común

            La interfaz común tiene dos niveles, y conviene distinguirlos. El
            **contrato de resultados** (`SearchHit` normalizado + `store.search`, con
            el filtro como parámetro) lo usan *todos* los consumidores: la CLI, los
            scripts de evaluación y estos notebooks. Sobre él, `DiscoveryService` es
            la **fachada de texto libre**: aplica el prefijo `query:` del contrato
            E5, codifica y delega en la base — es lo que usa la CLI
            (`make search q="..."`). Los scripts de evaluación, en cambio, entran
            por `store.search` con los **embeddings precomputados** de
            `data/embeddings/` — deliberadamente: así los artefactos entregados no
            dependen de recodificar las consultas en cada ejecución y son
            reproducibles bit a bit. Ambos caminos convergen en el mismo contrato.
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
            pd.DataFrame([hit.as_dict() for hit in hits])[
                ["rank", "product_id", "title", "brand", "native_score", "score_kind", "higher_is_better"]
            ].assign(title=lambda frame: frame["title"].str.slice(0, 60))
            """
        ),
        code(
            r"""
            branded_hits = service.search_text(
                "zapatillas cómodas para salir a correr", top_k=5, brand="NIKE"
            )
            assert {hit.brand for hit in branded_hits} == {"NIKE"}
            pd.DataFrame(
                [
                    {"rank": hit.rank, "score": round(hit.native_score, 4),
                     "brand": hit.brand, "title": hit.title[:60]}
                    for hit in branded_hits
                ]
            )
            """
        ),
        markdown(
            r"""
            El resultado normalizado **conserva el score nativo y su semántica**.
            Este detalle importa más de lo que parece: cada motor vectorial devuelve
            su score en una moneda distinta — Qdrant con métrica coseno devuelve una
            *similitud* (mayor es mejor), pero Chroma o Weaviate devuelven una
            *distancia* (menor es mejor), a veces bajo la misma etiqueta «cosine».
            Convertir una en otra «a ojo», o comparar scores de motores distintos
            como si fueran equivalentes, produce rankings sin sentido. Por eso el
            contrato lleva el score acompañado de su declaración
            (`score_kind="similarity"`, `higher_is_better=true`) en lugar de un
            número desnudo.

            ## Filtro de marca dentro de la consulta: prefiltrar, no tachar

            Hay dos maneras de combinar una búsqueda vectorial con una condición de
            metadatos, y no son equivalentes:

            - **Post-filtrado**: recuperar el top-k global y tachar lo que no cumple.
              Barato, pero roto por diseño — si solo 3 de los 10 mejores globales son
              de la marca, se devuelven 3 resultados (o ninguno), aunque el catálogo
              tenga cientos de productos de esa marca perfectamente válidos.
            - **Prefiltrado (filtro en la consulta)**: la condición entra en el
              motor, que restringe el universo de búsqueda a los puntos que cumplen
              el filtro y devuelve el top-k *de ese universo*. Siempre hay k
              resultados si existen k candidatos.

            El enunciado exige lo segundo, y el índice de payload *keyword* sobre
            `brand` (notebook 02) es lo que lo hace eficiente: sin él, el motor
            tendría que escanear payloads uno a uno. En Qdrant la condición viaja
            como `query_filter` de `query_points`, y el grafo HNSW la tiene en
            cuenta durante la navegación. Las cuatro consultas oficiales,
            verificadas en vivo con sus embeddings precomputados:
            """
        ),
        code(
            r"""
            from aurum_discovery import load_filtered_queries

            filtered_queries = load_filtered_queries()
            filtered_matrix = embedding_set.matrix("consultas_filtradas")
            filtered_ids = embedding_set.identifiers["consultas_filtradas"]
            verification_rows = []
            for _, query in filtered_queries.iterrows():
                vector = filtered_matrix[filtered_ids.index(query["workload_id"])]
                hits = store.search(vector, top_k=10, brand=query["filter_value"])
                brands = {hit.brand for hit in hits}
                assert brands == {query["filter_value"]}, (query["workload_id"], brands)
                verification_rows.append(
                    {"workload_id": query["workload_id"],
                     "consulta": query["query_text"][:42],
                     "marca": query["filter_value"],
                     "resultados": len(hits),
                     "todas_cumplen": brands == {query["filter_value"]}}
                )
            pd.DataFrame(verification_rows)
            """
        ),
        markdown(
            r"""
            ### El filtro cambia el universo, no recorta la lista

            La comparación siguiente lo demuestra: para la primera consulta filtrada,
            el top-10 **global** solo contiene algunos productos de la marca; el top-10
            **filtrado** devuelve diez productos de la marca, incluidos varios que el
            post-filtrado habría perdido.
            """
        ),
        code(
            r"""
            first_query = filtered_queries.iloc[0]
            vector = filtered_matrix[filtered_ids.index(first_query["workload_id"])]
            global_hits = store.search(vector, top_k=10)
            filtered_hits = store.search(vector, top_k=10, brand=first_query["filter_value"])
            surviving = [hit for hit in global_hits if hit.brand == first_query["filter_value"]]
            recovered = {hit.product_id for hit in filtered_hits} - {hit.product_id for hit in surviving}
            print(f"Consulta: {first_query['query_text']!r} · marca {first_query['filter_value']}")
            print(f"Top-10 global: {len(surviving)}/10 cumplen la marca")
            print(f"Top-10 filtrado: 10/10 cumplen; {len(recovered)} productos que el "
                  "post-filtrado habría perdido")
            """
        ),
        markdown(
            r"""
            ## Casos límite, uno a uno

            **Filtro sin resultados** — lista vacía sin excepción (es una respuesta
            válida, no un error):
            """
        ),
        code(
            r"""
            no_results = store.search(filtered_matrix[0], top_k=10, brand="MarcaInexistenteAurum")
            print(f"Marca inexistente -> {no_results!r}")
            """
        ),
        markdown(
            r"""
            **Colección vacía** — error accionable, no una lista vacía engañosa. Se
            demuestra con una colección efímera del propio namespace de la actividad,
            que se elimina al terminar con el token de confirmación exacto:
            """
        ),
        code(
            r"""
            from aurum_discovery import EmptyCollectionError

            empty_store = CatalogVectorStore(
                url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                collection_name="aurum-market-eval-demo-vacia",
                vector_size=embedding_set.configuration.dimension,
                hnsw=run_config.hnsw,
                ef_search=run_config.ef_search,
            )
            empty_store.ensure_collection()
            try:
                empty_store.search(filtered_matrix[0], top_k=5)
            except EmptyCollectionError as error:
                print(f"EmptyCollectionError: {error}")
            finally:
                empty_store.delete_collection(
                    confirmation="DELETE:aurum-market-eval-demo-vacia"
                )
            print("Colección efímera eliminada.")
            """
        ),
        markdown(
            r"""
            **Proveedor no disponible** — toda operación (búsqueda, ingesta, lecturas,
            borrados) traduce el fallo de transporte en `VectorStoreUnavailableError`
            con instrucciones, en lugar de propagar una excepción críptica del SDK.
            Se ejercitan las dos rutas que importan en este capítulo — el chequeo de
            conexión y la propia búsqueda:
            """
        ),
        code(
            r"""
            from aurum_discovery import VectorStoreUnavailableError

            unreachable_store = CatalogVectorStore(
                url="http://localhost:9999",
                collection_name="aurum-market-eval-inexistente",
                vector_size=embedding_set.configuration.dimension,
                hnsw=run_config.hnsw,
                ef_search=run_config.ef_search,
                timeout_seconds=2.0,
            )
            for operation_name, operation in (
                ("ping", lambda: unreachable_store.ping()),
                ("search", lambda: unreachable_store.search(filtered_matrix[0], top_k=5)),
            ):
                try:
                    operation()
                except VectorStoreUnavailableError as error:
                    print(f"{operation_name} -> VectorStoreUnavailableError: {str(error)[:90]}...")
            """
        ),
        markdown(
            r"""
            → Continúa en `actividad_04_operaciones_y_duplicados.ipynb`.
            """
        ),
    ]
