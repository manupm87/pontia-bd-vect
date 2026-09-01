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

            La familia utilizada es **HNSW** (*Hierarchical Navigable Small World*):
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

            ## Por qué Qdrant — y por qué la familia es HNSW

            El enunciado no impone motor, pero sí impone requisitos que discriminan
            entre los cinco que propone: filtro de metadatos **ejecutado por la base
            de datos**, fidelidad ANN medida contra un oráculo exacto, configuración
            ANN explícita, persistencia con reconstrucción desde cero y evaluación
            local sin costes para quien corrige. La comparativa *medida* entre
            motores queda explícitamente fuera («no construir un ranking entre nubes
            y local»), así que la elección se argumenta por criterios, no por
            cronómetro:

            | Criterio del enunciado | Qdrant | Chroma | Weaviate | Milvus | Pinecone |
            |---|---|---|---|---|---|
            | Parámetros HNSW explícitos | por colección **y por consulta** | vía metadatos | sí | sí | ocultos |
            | Oráculo exacto sobre la misma colección | `exact=true` por consulta | no | índice `flat` aparte | índice `FLAT` aparte | no |
            | Filtro en el plan de búsqueda | índice de payload | sí | sí | sí | sí |
            | Observabilidad del índice | `indexed_vectors_count` | limitada | sí | sí | opaca |
            | IDs UUID nativos | sí | cadenas | sí | enteros/cadenas | cadenas |
            | Despliegue para el corrector | 1 contenedor | embebido | 1 contenedor + módulos | compose de 3 servicios | solo cloud |

            Dos criterios acabaron decidiendo. El **oráculo exacto por consulta sobre
            la misma colección** hace medible la fidelidad ANN sin montar un segundo
            sistema — es la columna vertebral de este notebook y del 05 —, y la
            **observabilidad del índice** permitió cazar la lección del índice
            fantasma que se cuenta más abajo (con Chroma, que es aún más simple de
            desplegar, ese estado interno es invisible). Pinecone quedó descartado
            además por ocultar la decisión de índice — el enunciado obliga entonces a
            explicar qué control se pierde, y aquí se quería ejercer ese control, no
            renunciar a él —; Milvus es el único que habría permitido comparar
            familias (IVF, PQ…) dentro del mismo motor, pero paga ese poder con una
            huella operativa difícil de justificar para 15.000 productos en un
            portátil.

            Elegido Qdrant, **la familia ANN deja de ser una elección: Qdrant solo
            implementa HNSW**. Que esa restricción no duela se argumenta con lo
            aprendido en la sesión 02, donde se trabajaron Flat, IVF, PQ y HNSW
            sobre FAISS:

            - **IVF** exige una fase de `train` (k-means sobre una muestra) y pierde
              recall en las fronteras entre clústeres. Peor aún para este caso: las
              **altas incrementales** del control de catálogo (notebook 04) irían
              degradando unos centroides congelados hasta forzar re-entrenamientos
              periódicos. HNSW inserta vector a vector sin fase de entrenamiento —
              exactamente el patrón de escritura de este sistema.
            - **PQ** comprime a costa de distorsionar los scores, y aquí el score
              coseno crudo es señal de negocio: el umbral de duplicados (0.943,
              notebook 04) vive en un hueco de ~0.09 que el ruido de cuantización
              se comería. Con 15.000×1024d en float32 (~60 MB) la compresión
              tampoco compra nada todavía.
            - **Flat** no se pierde: es el modo exacto de la propia colección, y
              este notebook lo usa como oráculo unas celdas más abajo.

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
            | Vector | 1024d float32, L2-normalizado | contrato del modelo E5 (large) |
            | Métrica | coseno | score = similitud, mayor es mejor |
            | Payload | product_id, title, brand, color, locale, catalog_version, active | filtros y presentación |
            | Índice de payload | `brand` (keyword) | filtro dentro del plan de búsqueda |
            | Nulos | cadena vacía, siempre | "vacío ≠ 'nan'", criterio único |
            | HNSW | m=24, ef_construct=120, ef_search=128 | única familia del motor; justificada arriba |

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
            ## Una medición honesta: dónde empieza a ganar el índice

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
            La lectura honesta de esta medición es que **este catálogo está justo en
            la frontera donde indexar empieza a compensar** — y la frontera se mueve
            con $N \times d$. Con 384 dimensiones (e5-small), el escaneo
            secuencial de 15.000×384 floats era tan barato y tan amigo de la caché
            que empataba o incluso ganaba al grafo, cuyo coste fijo (saltos por
            capas, cola de candidatos con `ef=128`) no se amortizaba. Con las
            1024 dimensiones de la configuración final el coste del escaneo casi
            se triplica, el del grafo apenas cambia, y el HNSW ya araña décimas
            de milisegundo al exhaustivo. Lo que de verdad compra el índice es la
            **curva de crecimiento**: el escaneo es $O(N \cdot d)$ y se
            multiplica por ~67 al pasar a un millón de productos, mientras que la
            búsqueda HNSW crece de forma aproximadamente logarítmica. La decisión
            de indexar no se toma por la foto de hoy sino por la pendiente;
            tenerla medida — y no supuesta — es precisamente lo que este notebook
            deja registrado. El notebook 05 completa la figura con la otra cara:
            cuánta fidelidad pierde el índice a cambio (ninguna, con
            `ef_search=128`).

            ## La latencia que ve el usuario: el encoder manda

            Los milisegundos anteriores son solo la mitad de la historia. Una
            consulta interactiva de la CLI paga dos veces: **codificar** el texto
            con el modelo y **buscar** en la colección. Al promover un modelo de
            1024d y ~560M de parámetros conviene medir cuánto pesa cada mitad,
            con el mismo protocolo:
            """
        ),
        code(
            r"""
            from aurum_discovery import encode_texts, load_encoder, load_evaluation_queries

            encoder = load_encoder(embedding_set.configuration.model_id)
            query_texts = load_evaluation_queries()["query_text"].tolist()
            encode_report = measure_latency(
                [
                    (lambda text=text: encode_texts(
                        encoder, [text],
                        prefix=embedding_set.configuration.query_prefix,
                        normalize=embedding_set.configuration.normalize,
                    ))
                    for text in query_texts
                ],
                warmup=run_config.latency_warmup,
                repeats=run_config.latency_repeats,
            )
            print(f"Codificar 1 consulta: p50 {encode_report.p50_ms:.1f} ms, "
                  f"p95 {encode_report.p95_ms:.1f} ms")
            print(f"Buscar (HNSW ef=128): p50 {latency_modes.iloc[1]['p50_ms']} ms")
            """
        ),
        markdown(
            r"""
            Incluso con GPU, codificar la consulta cuesta más del doble que
            buscarla en Qdrant — y en CPU la diferencia se multiplica. Ese es el
            precio real de la promoción a e5-large: no lo paga el índice (que
            crece linealmente en memoria con $d$ y apenas en tiempo), lo paga
            cada consulta al codificarse. Para una CLI de descubrimiento
            interactiva sigue siendo imperceptible; en un servicio de alto QPS
            el encoder sería la primera cifra a vigilar, y la validación
            ampliada del notebook 01 sería el argumento para decidir si la
            calidad extra lo amortiza.

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
              ingest`, ~8 s) porque los embeddings están persistidos con manifiesto; el
              interruptor de reset está **desactivado por defecto** para que ninguna
              ejecución rutinaria pueda recrear la colección por accidente.
            - El esquema se re-verifica en cada arranque (`_assert_schema`): dimensión o
              métrica distintas abortan con un error accionable en español.

            → Continúa en `actividad_03_recuperacion_y_filtros.ipynb`.
            """
        ),
    ]
