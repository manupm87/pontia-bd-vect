"""Notebook 02: schema, explicit ANN configuration, ingestion, persistence."""

from __future__ import annotations

from nbformat import NotebookNode

from .common import code, markdown, setup_cells, store_cells

FILENAME = "actividad_02_indice_y_bbdd.ipynb"


def build_cells() -> list[NotebookNode]:
    return [
        markdown(
            r"""
            # 02 · Índice y base de datos vectorial

            **Bloque 3** de los criterios: esquema, configuración ANN explícita,
            ingesta idempotente, persistencia y verificación del estado antes de
            aceptar consultas. Motor elegido: **Qdrant v1.18** en Docker, con su SDK
            nativo (`qdrant-client`); LangChain no aparece en el núcleo, tal y como
            pide el enunciado.

            ## De la fuerza bruta al vecino aproximado (ANN)

            El oráculo del notebook 01 compara la consulta contra los 15.000
            vectores: coste $O(N \cdot d)$ por consulta. A esta escala son
            milisegundos; a millones de productos, no. Los índices **ANN**
            (*Approximate Nearest Neighbor*) responden «casi el mismo top-k» en una
            fracción del tiempo, a cambio de poder perder algún vecino verdadero —
            una pérdida que en el notebook 05 se medirá con nombre propio
            (*fidelidad*).

            La familia elegida es **HNSW** (*Hierarchical Navigable Small World*):
            un grafo donde cada vector se conecta con sus `m` vecinos más próximos,
            organizado en capas jerárquicas — las superiores, poco pobladas, permiten
            saltos largos; las inferiores afinan localmente. Buscar es navegar el
            grafo por *greedy search* hacia la consulta. Tres parámetros gobiernan el
            compromiso calidad/coste:

            | Parámetro | Qué controla | Aquí |
            |---|---|---|
            | `m` | conexiones por nodo: más aristas, mejor recall y más memoria | 24 |
            | `ef_construct` | amplitud de la búsqueda **al construir** el grafo: calidad de las aristas | 120 |
            | `ef_search` | amplitud de la búsqueda **al consultar**: candidatos vivos por salto | 128 |

            `m` y `ef_construct` se pagan una vez (construcción); `ef_search` se paga
            en cada consulta y es el mando de ajuste fino en producción.

            ## Qué añade la base de datos sobre el índice

            Un índice (FAISS, por ejemplo) es una estructura en memoria; una **base
            de datos vectorial** lo envuelve con lo que un sistema vivo necesita. En
            el vocabulario de Qdrant:

            - **Colección**: el conjunto de vectores con un esquema común
              (dimensión, métrica de distancia, configuración del índice).
            - **Punto**: la unidad almacenada — un ID, un vector y un **payload**
              (los metadatos JSON del producto: marca, título, versión…).
            - **Índice de payload**: un índice invertido clásico sobre un campo del
              payload (aquí `brand`, tipo *keyword*), que permite filtrar **dentro**
              de la búsqueda vectorial en vez de después.
            - **Persistencia**: los puntos viven en disco (un volumen Docker), no en
              la memoria de un proceso: sobreviven a reinicios y se pueden
              reconstruir desde cero.

            ## El esquema, decidido y verificable

            | Decisión | Valor | Por qué |
            |---|---|---|
            | ID de punto | `record_id` (UUIDv5 del `product_id`) | idempotencia estructural: reingerir = upsert |
            | Vector | 384d float32, L2-normalizado | contrato del modelo E5 |
            | Métrica | coseno | score = similitud, mayor es mejor |
            | Payload | product_id, title, brand, color, locale, catalog_version, active | filtros y presentación |
            | Índice de payload | `brand` (keyword) | filtro dentro del plan de búsqueda |
            | Nulos | cadena vacía, siempre | "vacío ≠ 'nan'", criterio único |
            | HNSW | m=24, ef_construct=120, ef_search=128 | familia trabajada en la sesión 02 |

            La primera fila esconde la decisión más rentable del sistema. Un
            **UUIDv5** es un identificador *determinista*: se calcula aplicando un
            hash al `product_id` dentro de un espacio de nombres fijo, así que el
            mismo producto produce **siempre** el mismo ID, en cualquier máquina y
            en cualquier reejecución. Combinado con la semántica *upsert* de la base
            (insertar-o-sobrescribir por ID), la ingesta se vuelve **idempotente**
            por construcción: repetirla no puede duplicar registros, porque
            reescribir el mismo ID es una actualización, no una inserción. La
            idempotencia — que ejecutar una operación dos veces deje el mismo estado
            que ejecutarla una — no depende aquí de la disciplina de quien opera,
            sino del diseño del identificador.
            """
        ),
        *setup_cells(),
        *store_cells(),
        markdown(
            r"""
            ## La lección del índice fantasma

            Qdrant no guarda la colección como un bloque: la trocea en **segmentos**,
            y un proceso de **optimizadores** en segundo plano los fusiona y decide
            cuándo construir el grafo HNSW de cada uno. La configuración HNSW puede
            estar perfectamente escrita **y no existir**: Qdrant solo construye el
            grafo cuando un segmento supera `indexing_threshold` (10–20 MB por
            defecto), y este catálogo repartido en segmentos no lo alcanzaba.
            Resultado: `status=green`,
            `indexed_vectors_count=0`, cada búsqueda era un escaneo exhaustivo y
            `ef_search` no hacía nada. Una revisión adversarial lo destapó comprobando
            la colección viva. Desde entonces la creación fija `indexing_threshold=100`
            KB y **la verificación de arranque exige vectores indexados, no el
            semáforo verde**:
            """
        ),
        code(
            r"""
            from qdrant_client import models

            index_state = store.wait_until_indexed()
            print(f"Estado: {index_state}")
            assert index_state["indexed_vectors_count"] >= 0.8 * index_state["points_count"], index_state

            info = store.collection_info()
            vector_config = info.config.params.vectors
            assert vector_config.size == embedding_set.configuration.dimension, vector_config
            assert vector_config.distance == models.Distance.COSINE, vector_config
            assert info.payload_schema["brand"].data_type == models.PayloadSchemaType.KEYWORD
            print(f"Vector: {vector_config.size}d · distancia {vector_config.distance}")
            print(f"HNSW: m={info.config.hnsw_config.m}, "
                  f"ef_construct={info.config.hnsw_config.ef_construct}")
            print(f"indexing_threshold: {info.config.optimizer_config.indexing_threshold} KB")
            print(f"Índices de payload: {list(info.payload_schema)}")
            """
        ),
        code(
            r"""
            from aurum_discovery import load_catalog

            catalog = load_catalog()
            sample_record_id = catalog.iloc[0]["record_id"]
            payload = store.retrieve(sample_record_id)
            expected_fields = {
                "product_id", "title", "brand", "color", "locale",
                "catalog_version", "active",
            }
            assert payload is not None and set(payload) == expected_fields, payload
            print(f"Payload de un punto real: {sorted(payload)}")
            """
        ),
        markdown(
            r"""
            > ¡Ojo! `indexed_vectors_count` puede superar temporalmente a
            > `points_count`: Qdrant contabiliza los
            > vectores indexados **por segmento**, y mientras los optimizadores
            > fusionan segmentos el mismo punto puede estar contado en el segmento
            > viejo y en el nuevo. Es un artefacto transitorio de la contabilidad
            > interna, no puntos fantasma — por eso la verificación exige cobertura
            > (`>= 80 %` de los puntos, aquí 100 %), no igualdad exacta.
            """
        ),
        markdown(
            r"""
            ## Una medición honesta: a esta escala, el índice todavía no gana

            La misma colección responde en modo exacto (`SearchParams.exact=true`, el
            oráculo) y en modo HNSW. Medimos ambas rutas con el protocolo compartido
            del proyecto (`measure_latency`, con el calentamiento y las repeticiones
            de `config/run_config.yaml`):
            """
        ),
        code(
            r"""
            from aurum_discovery import measure_latency

            evaluation_matrix = embedding_set.matrix("consultas_evaluacion")
            rows = []
            for mode_name, exact in (("exacto (oráculo)", True), ("HNSW ef=128", False)):
                operations = [
                    (lambda vector=evaluation_matrix[row], exact=exact:
                     store.search(vector, top_k=10, exact=exact))
                    for row in range(evaluation_matrix.shape[0])
                ]
                report = measure_latency(
                    operations, warmup=run_config.latency_warmup,
                    repeats=run_config.latency_repeats,
                )
                rows.append(
                    {"modo": mode_name,
                     "p50_ms": round(report.p50_ms, 2),
                     "p95_ms": round(report.p95_ms, 2)}
                )
            latency_modes = pd.DataFrame(rows)
            latency_modes
            """
        ),
        code(
            r"""
            import plotly.express as px

            px.bar(
                latency_modes.melt(id_vars="modo", var_name="percentil", value_name="ms"),
                x="modo", y="ms", color="percentil", barmode="group",
                title="Latencia por consulta: escaneo exhaustivo frente a HNSW (misma colección)",
            )
            """
        ),
        markdown(
            r"""
            La lectura honesta de esta medición es que **a 15.000 vectores el índice
            no aporta ventaja de latencia** — ambas rutas responden en ~2-3 ms y el
            HNSW puede incluso salir un pelo más lento: recorrer un grafo por capas
            tiene un coste fijo (saltos, colas de candidatos con `ef=128`) que un
            escaneo secuencial de 15.000×384 floats — un recorrido trivial y muy
            amigo de la caché — todavía no amortiza. Lo que compra el índice es la
            **curva de crecimiento**: el escaneo es $O(N)$ y se multiplica por ~67 al
            pasar a un millón de productos, mientras que la búsqueda HNSW crece de
            forma aproximadamente logarítmica. La decisión de indexar no se toma por
            la foto de hoy sino por la pendiente; tenerla medida — y no supuesta — es
            precisamente lo que este notebook deja registrado. El notebook 05
            completa la figura con la otra cara: cuánta fidelidad pierde el índice a
            cambio (ninguna, con `ef_search=128`).

            ## Ingesta por lotes e idempotente

            La ingesta oficial (`make ingest`) sube lotes de 256 con `wait=True` y
            verifica recuento e indexación antes de dar la colección por buena. La
            prueba de idempotencia en vivo: reingerimos un lote y el recuento no se
            mueve, porque el ID de punto es el UUIDv5 estable del producto.
            """
        ),
        code(
            r"""
            from aurum_discovery import iter_record_batches

            count_before = store.count()
            first_batch = next(
                iter_record_batches(catalog, embedding_set.matrix("products"), batch_size=256)
            )
            store.upsert_records([[record for record in first_batch]])
            assert store.count() == count_before
            print(f"Reingesta de 256 registros: {count_before} -> {store.count()} (sin duplicados)")
            """
        ),
        code(
            r"""
            ingest_report = json.loads(
                (project_root / ".artifacts" / "ingesta" / "informe_ingesta.json").read_text()
            )
            ingest_report
            """
        ),
        markdown(
            r"""
            ## Persistencia y reconstrucción desde cero

            - La colección vive en el volumen Docker `aurum-market-eval-qdrant-data`:
              sobrevive a reinicios del contenedor (`make down && make up`).
            - La reconstrucción completa es un comando (`AURUM_ALLOW_RESET=true make
              ingest`, ~5 s) porque los embeddings están persistidos con manifiesto; el
              interruptor de reset está **desactivado por defecto** para que ninguna
              ejecución rutinaria pueda recrear la colección por accidente.
            - El esquema se re-verifica en cada arranque (`_assert_schema`): dimensión o
              métrica distintas abortan con un error accionable en español.

            → Continúa en `actividad_03_recuperacion_y_filtros.ipynb`.
            """
        ),
    ]
