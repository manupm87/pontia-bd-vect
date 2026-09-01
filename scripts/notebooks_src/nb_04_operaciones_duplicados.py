"""Notebook 04: catalog events, visibility, and the duplicate-control rule."""

from __future__ import annotations

from nbformat import NotebookNode

from .common import code, markdown, setup_cells, store_cells

FILENAME = "actividad_04_operaciones_y_duplicados.ipynb"


def build_cells() -> list[NotebookNode]:
    return [
        markdown(
            r"""
            # 04 · Operaciones y control de catálogo

            **§4 del enunciado**: aplicar los 24 eventos respetando `sequence` con un
            proceso repetible, medir la visibilidad de cada tipo de operación por ID y
            por búsqueda, y diseñar una regla reproducible de detección de altas
            duplicadas calibrada **solo** con desarrollo.

            ## 4.1 Eventos: idempotencia y visibilidad

            Dos conceptos gobiernan esta sección. La **idempotencia** ya quedó
            definida en el notebook 02 (repetir la operación deja el mismo estado);
            aquí se exige a la secuencia completa de eventos, no solo a la ingesta.
            La **visibilidad** es la otra cara de escribir en un sistema real: que el
            motor *confirme* una escritura no garantiza que una lectura inmediata la
            *vea* — entre medias puede haber colas, réplicas o índices pendientes
            (los sistemas distribuidos llaman a esto *consistencia eventual*). Un
            sistema serio no asume la visibilidad: la **mide** por cada ruta de
            lectura (lectura por ID y búsqueda vectorial) y sabe esperar con un
            *deadline* — aquí, un sondeo acotado (`wait_until`) que reintenta hasta
            observar el estado esperado o falla con un timeout explícito. Qdrant con
            `wait=True` confirma la escritura ya aplicada, así que veremos converger
            todo al primer intento; el mecanismo de espera existe igualmente, porque
            el contrato del sistema no debe depender de la suerte del motor elegido.

            Los eventos oficiales — que alteran el catálogo de forma permanente — se
            aplican con `make events` sobre una colección dedicada
            (`aurum-market-eval-eventos`) sembrada con el catálogo íntegro, para que
            la colección principal siga sirviendo los artefactos reproducibles. La
            demo del canario de más abajo sí escribe sobre la colección principal,
            pero con un registro propio que **siempre** se elimina al terminar: al
            acabar la celda, la colección queda exactamente como estaba. El proceso
            de eventos aplica la secuencia **dos veces**, exige el mismo estado
            final, y verifica ese estado registro a registro (no solo el recuento):
            los 24 record_id afectados se leen por ID y deben mostrar la versión
            entregada o haber dejado de existir.
            """
        ),
        *setup_cells(),
        *store_cells(),
        code(
            r"""
            from aurum_discovery import load_catalog, load_catalog_events

            events = load_catalog_events()
            catalog = load_catalog()
            catalog_ids = set(catalog["record_id"])
            events["tipo"] = [
                "actualización" if op == "UPSERT" and rid in catalog_ids
                else "alta" if op == "UPSERT"
                else "eliminación"
                for op, rid in zip(events["operation"], events["record_id"], strict=True)
            ]
            events["tipo"].value_counts().rename_axis("tipo").reset_index(name="eventos")
            """
        ),
        code(
            r"""
            events_report = json.loads(
                (project_root / ".artifacts" / "eventos" / "informe_eventos.json").read_text()
            )
            print(f"Recuento esperado tras 8 altas y 8 bajas: {events_report['expected_final_count']}")
            print(f"Tras la 1ª pasada: {events_report['count_after_first_pass']} · "
                  f"tras la 2ª: {events_report['count_after_second_pass']} · "
                  f"idempotente: {events_report['idempotent']}")
            print(f"Estado final registro a registro: {events_report['final_state_checks']}")
            pd.DataFrame(events_report["visibility"])
            """
        ),
        markdown(
            r"""
            La tabla de visibilidad demuestra lo que pide el enunciado — no cronometrar
            proveedores, sino verificar que **una escritura confirmada acaba siendo
            observable**: la actualización muestra `catalog_version=2` leyendo por ID y
            aparece en el top-3 de su propio vector; el alta es recuperable por ambas
            rutas; la baja deja de serlo. La espera es activa y acotada (`wait_until`
            con deadline): el sistema sabe esperar, y sabría fallar con un timeout
            explícito si la visibilidad no llegara.

            El ciclo completo, en vivo, con un **registro canario**: un dato de
            prueba con ID propio y reconocible que se inserta solo para verificar el
            comportamiento del sistema y se elimina al terminar — como el canario de
            las minas, avisa sin costar nada. El borrado va en un `finally`, así que
            la celda es re-ejecutable incluso si una verificación fallara:
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
        markdown(
            r"""
            ## 4.2 Altas potencialmente duplicadas

            Detectar duplicados es la segunda cara del mismo espacio vectorial: si
            dos fichas describen el mismo producto — aunque cambien el orden del
            título, pierdan la marca o recorten la descripción — sus embeddings caen
            casi en el mismo punto, y su similitud coseno (notebook 01) se dispara
            hacia 1. La regla explota esa geometría y es deliberadamente simple y
            reproducible: la base vectorial genera los 5 candidatos más próximos (el
            alta se codifica como *documento*, con la misma composición y prefijo
            `passage:` que el catálogo — es una comparación documento-documento, no
            consulta-documento) y se declara duplicado si el mejor candidato alcanza
            un **umbral** de similitud. Como rasgo adicional se registra el
            **margen**: la ventaja del mejor candidato sobre el segundo — un
            duplicado real suele destacar sobre el resto, mientras que un producto
            genérico tiene muchos vecinos igual de parecidos.

            **Calibrar** la regla es elegir ese umbral con datos etiquetados, y
            conlleva un riesgo con nombre: **sobreajuste** — clavar el parámetro a
            las particularidades de los 14 ejemplos de desarrollo y que no
            generalice. Por mandato del enunciado (y buena práctica), la calibración
            usa **exclusivamente** `altas_desarrollo.csv`; el conjunto ciego no se
            mira hasta tener la regla congelada.

            La distribución de scores en desarrollo lo dice casi todo:
            """
        ),
        code(
            r"""
            calibration = json.loads(
                (project_root / ".artifacts" / "duplicados" / "calibracion.json").read_text()
            )
            calibration_cases = pd.DataFrame(
                [
                    {"incoming_id": case["incoming_id"],
                     "etiqueta": "duplicado" if case["is_duplicate"] else "no duplicado",
                     "score_top1": case["top_candidates"][0]["native_score"],
                     "margen": case["top_candidates"][0]["native_score"]
                     - case["top_candidates"][1]["native_score"]}
                    for case in calibration["cases"]
                ]
            )
            calibration_cases.round(4)
            """
        ),
        code(
            r"""
            import plotly.express as px

            threshold = run_config.duplicate_score_threshold
            figure = px.strip(
                calibration_cases, x="etiqueta", y="score_top1", color="etiqueta",
                hover_data=["incoming_id"],
                title="Score del mejor candidato en desarrollo: separación y umbral",
            )
            figure.add_hline(y=threshold, line_dash="dash",
                             annotation_text=f"umbral {threshold}")
            figure.update_layout(showlegend=False)
            figure
            """
        ),
        markdown(
            r"""
            Los 7 duplicados puntúan en [0.988, 1.000] y los 7 no duplicados en
            [0.874, 0.898]: **cualquier umbral de score dentro del hueco separa
            perfectamente** el desarrollo (con margen exigido 0). El hueco mide
            0.090 de score — casi una décima entera de similitud coseno — y
            desde esta ejecución queda persistido en el artefacto de calibración
            (`separation`), que además aborta si el umbral configurado se sale
            de él. La rejilla explora 104 combinaciones de umbral × margen y
            deja dos lecciones:

            - Sobre el **score**, todo umbral en el hueco alcanza F1=1.0. Elegir el
              extremo superior que devuelve la rejilla (0.9883) sería pegarse al
              duplicado más débil, así que la configuración final fija un valor
              centrado (**0.943**, punto medio 0.9431; criterio *max-margin*:
              maximizar la distancia de la frontera al ejemplo más cercano de cada
              clase, la misma idea que hace robustas a las máquinas de vectores
              soporte).
            - Sobre el **margen**, la rejilla lo descarta activamente: exigir margen
              ≥ 0.005 ya rompe la separación perfecta, porque hay duplicados reales
              casi empatados con un segundo candidato (DEV-DUP-001 y DEV-DUP-002
              ganan a su siguiente vecino por solo 0.004 — el catálogo contiene
              productos legítimamente casi idénticos entre sí). El margen queda
              registrado como evidencia por caso, pero con umbral 0: no filtra.

            ### Cómo se puntúa un clasificador binario

            Cada decisión cae en una de cuatro casillas según predicción y realidad
            (la **matriz de confusión**): verdadero positivo (TP), falso positivo
            (FP), falso negativo (FN) y verdadero negativo (TN). De ahí salen las
            tres métricas del enunciado:

            $$\text{precision} = \frac{TP}{TP + FP} \qquad \text{recall} = \frac{TP}{TP + FN} \qquad F_1 = 2\cdot\frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$$

            La *precision* responde «de lo que marqué como duplicado, ¿cuánto lo
            era?»; el *recall*, «de los duplicados reales, ¿cuántos cacé?»; la $F_1$
            es su media armónica, que castiga descompensar una a costa de la otra.

            > ¡Ojo! Aquí un TP exige algo más que acertar la clase: la predicción
            > positiva debe señalar **el product_id de referencia exacto**. Marcar
            > "duplicado" apuntando al producto equivocado cuenta como error, porque
            > en negocio esa revisión también falla.
            """
        ),
        code(
            r"""
            configured_rule = calibration["configured_rule"]
            print(f"Regla configurada: score >= {run_config.duplicate_score_threshold} "
                  f"(margen >= {run_config.duplicate_margin_threshold})")
            print(f"Desarrollo -> precision {configured_rule['precision']:.3f} · "
                  f"recall {configured_rule['recall']:.3f} · F1 {configured_rule['f1']:.3f}")
            print(f"(TP={configured_rule['true_positive']}, FP={configured_rule['false_positive']}, "
                  f"FN={configured_rule['false_negative']}, TN={configured_rule['true_negative']}; "
                  "un TP exige señalar el product_id de referencia exacto)")
            """
        ),
        markdown(
            r"""
            **Falsos positivos y falsos negativos no cuestan lo mismo.** Un FP retiene
            una ficha legítima en revisión humana: coste visible, acotado y reversible.
            Un FN publica un duplicado: fragmenta reseñas y stock y degrada el propio
            buscador — coste mayor y silencioso. En desarrollo no hay ni unos ni
            otros; el caso a vigilar es `DEV-NEW-007` (no duplicado, score 0.898, a
            0.045 del umbral): productos casi idénticos de catálogo legítimo son el
            modo de fallo natural de esta regla al crecer el catálogo.

            ## Decisión sobre el conjunto ciego

            La regla congelada se aplica a `altas_evaluacion.csv` (`make duplicates`);
            toda predicción positiva señala su `product_id` candidato:
            """
        ),
        code(
            r"""
            blind_decisions = json.loads(
                (project_root / ".artifacts" / "duplicados" / "decisiones_evaluacion.json").read_text()
            )
            blind_table = pd.DataFrame(
                [
                    {"incoming_id": item["incoming_id"],
                     "predicción": item["decision"]["predicted_duplicate"],
                     "candidato": item["decision"]["matched_product_id"] or "—",
                     "score": round(float(item["decision"]["score"]), 4),
                     "margen": round(item["margin"], 4)}
                    for item in blind_decisions["decisions"]
                ]
            )
            blind_table
            """
        ),
        code(
            r"""
            blind_table["etiqueta"] = np.where(blind_table["predicción"], "positiva", "negativa")
            figure = px.strip(
                blind_table, x="etiqueta", y="score", color="etiqueta",
                hover_data=["incoming_id"],
                title="Conjunto ciego: la misma separación que en desarrollo",
            )
            figure.add_hline(y=run_config.duplicate_score_threshold, line_dash="dash")
            figure.update_layout(showlegend=False)
            figure
            """
        ),
        markdown(
            r"""
            Trece de los catorce casos replican la separación de desarrollo:
            positivos en [0.980, 1.000], negativos en [0.866, 0.889]. El
            decimocuarto es la excepción que enseña: `EVAL-DUP-004` puntúa
            **0.9428 — dentro del hueco de desarrollo y a 0.0002 del umbral** —
            y la regla lo clasifica negativo. Las etiquetas del ciego están
            reservadas, pero seamos francos con lo que el propio fichero
            delata: la convención de identificadores (`EVAL-DUP-*` frente a
            `EVAL-NEW-*`, la misma que en desarrollo) indica que este alta es
            **casi con certeza un duplicado real**, es decir, un probable
            **falso negativo** — el error caro. La lectura honesta del ciego
            es entonces precision estimada 6/6 y recall estimado 6/7 (~0.86),
            no una separación perfecta. Aun así el umbral no se toca: el
            conjunto de calibración no contiene ningún ejemplo en esa zona
            (ningún umbral del hueco tenía evidencia a favor o en contra) y
            moverlo *a posteriori* para cazar un caso del ciego sería calibrar
            con el conjunto de evaluación, exactamente lo que la metodología
            declaró que no haría. En producción este score intermedio es la
            definición operativa de «revisión humana»: la mejora de la regla no
            es otro umbral, sino una **banda de abstención** alrededor de la
            frontera que envíe estos casos a una cola de revisión en vez de
            decidirlos en automático.

            ## 4.3 Seguridad de las operaciones

            - Prefijo obligatorio `aurum-market-eval` validado antes de crear el
              cliente: imposible operar sobre una colección ajena por error.
            - `AURUM_ALLOW_RESET=false` y `AURUM_CONFIRM_CLEANUP` vacío por defecto;
              borrar exige el token exacto `DELETE:<colección>`.
            - Sin credenciales en el repositorio y sin servicios cloud en el recorrido
              evaluado.

            → Continúa en `actividad_05_evaluacion_y_analisis.ipynb`.
            """
        ),
    ]
