"""Validate challenger embeddings on an expanded ESCI query set.

The eight development queries are too few to promote a model with confidence:
a single query moves each macro metric by 0.125 of its per-query range. This
script rebuilds a much larger labeled workload from the *public* ESCI dataset
(the activity snapshot is derived from it, so product_ids and query_ids line
up), restricted to judged products that exist in the 15.000-product catalog.

Method and caveats, declared up front:
- Queries overlapping the activity are excluded: the 8 development query_ids
  and the 4 base query_ids behind the 12 blind evaluation paraphrases.
- Judgments are restricted to in-catalog products; unjudged retrieved
  products contribute zero gain, exactly as in the development experiments.
  Sparse judgments depress absolute numbers for every model equally, so only
  PAIRED comparisons against the incumbent are meaningful here.
- Two tiers are reported: "primaria" (>= 5 judged products in catalog) is
  the decision tier; "robustez" (>= 3) checks the ordering holds on a much
  larger but sparser set. Both require >= 1 E and >= 2 E/S in catalog.
- Rankings are exhaustive cosine over the persisted product matrices, like
  scripts/run_experiments.py: representation quality, no index effects.

The verdict for each challenger against the incumbent uses paired statistics:
mean per-query delta, a seeded bootstrap 95% CI, a sign-flip permutation
p-value, and win/tie/loss counts. Because a significance headline picked from
many contrasts must survive multiplicity, every family of contrasts also
carries Holm-adjusted p-values, and the head-to-head contrasts the report
cites (DIRECT_PAIRS) are computed and persisted here, never ad hoc.

The public examples parquet is NOT versioned (51 MB); download it once into
data/validacion/ as printed by the error message below.
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from aurum_discovery.config import (
    ARTIFACTS_DIRECTORY,
    PROJECT_ROOT,
    load_run_config,
)
from aurum_discovery.data import (
    EVALUATION_QUERIES_PATH,
    load_catalog,
    load_development_queries,
)
from aurum_discovery.embeddings import (
    EMBEDDING_CONFIGURATIONS,
    SET_PRODUCTS,
    encode_texts,
    load_embedding_set,
    load_encoder,
)
from aurum_discovery.evaluation import evaluate_query, macro_average
from aurum_discovery.operations import write_json_artifact

PARQUET_PATH = PROJECT_ROOT / "data" / "validacion" / "esci_examples.parquet"
PARQUET_URL = (
    "https://github.com/amazon-science/esci-data/raw/main/"
    "shopping_queries_dataset/shopping_queries_dataset_examples.parquet"
)
INCUMBENT = "e5_base_title"
# Contrastes directos entre los primeros clasificados: (referencia, retador).
# Se calculan y persisten aquí para que ninguna cifra del informe dependa de
# cuentas ad hoc no reproducibles.
DIRECT_PAIRS = (
    ("bge_m3_title", "e5_large_title"),
    ("qwen3_embedding_title", "e5_large_title"),
)
TIERS = {"primaria": 5, "robustez": 3}
BOOTSTRAP_RESAMPLES = 10_000
METRIC_ATTRIBUTES = ("ndcg_at_10", "recall_at_10", "mrr_at_10")


def load_public_judgments(parquet_path: Path) -> pd.DataFrame:
    """Load the Spanish ESCI examples, excluding the activity's queries."""
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"No existe {parquet_path}. Descárgalo una vez con:\n"
            f"  curl -L --create-dirs -o {parquet_path} {PARQUET_URL}"
        )
    examples = pd.read_parquet(
        parquet_path,
        columns=["query_id", "query", "product_id", "product_locale", "esci_label"],
    )
    examples = examples[examples["product_locale"] == "es"]
    development_ids = set(load_development_queries()["query_id"].astype(int))
    evaluation = pd.read_csv(EVALUATION_QUERIES_PATH)
    evaluation_ids = {
        int(identifier.split("-")[1]) for identifier in evaluation["evaluation_id"]
    }
    excluded = development_ids | evaluation_ids
    return examples[~examples["query_id"].isin(excluded)].copy()


def build_tiers(
    examples: pd.DataFrame, catalog_product_ids: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Restrict judgments to the catalog and select the tiered query sets.

    Returns the in-catalog judgments (with the activity's column contract),
    the deduplicated query texts, and one query frame per tier. Tiers are
    nested (robustez contains primaria), so texts are encoded once.
    """
    in_catalog = examples[examples["product_id"].isin(catalog_product_ids)].copy()
    counts = in_catalog.pivot_table(
        index="query_id",
        columns="esci_label",
        values="product_id",
        aggfunc="count",
        fill_value=0,
    )
    for label in ("E", "S"):
        if label not in counts:
            counts[label] = 0
    counts["judged"] = counts.sum(axis=1)
    eligible = counts[(counts["E"] >= 1) & (counts["E"] + counts["S"] >= 2)]

    tier_queries: dict[str, pd.DataFrame] = {}
    texts = (
        in_catalog.groupby("query_id")["query"].first().rename("query_text")
    ).reset_index()
    for tier_name, minimum_judged in TIERS.items():
        selected = eligible[eligible["judged"] >= minimum_judged].index
        tier_queries[tier_name] = (
            texts[texts["query_id"].isin(selected)]
            .sort_values("query_id")
            .reset_index(drop=True)
        )
    judgments = in_catalog.rename(columns={"query": "query_text"})[
        ["query_id", "product_id", "esci_label"]
    ].reset_index(drop=True)
    union_ids = set(tier_queries["robustez"]["query_id"])
    queries = texts[texts["query_id"].isin(union_ids)].reset_index(drop=True)
    # evaluate_query trabaja con query_id de tipo str, como en la actividad.
    judgments["query_id"] = judgments["query_id"].astype(str)
    queries["query_id"] = queries["query_id"].astype(str)
    for tier_frame in tier_queries.values():
        tier_frame["query_id"] = tier_frame["query_id"].astype(str)
    return judgments, queries, tier_queries


def _release_encoder(encoder: object) -> None:
    """Free the encoder's GPU memory before loading the next model."""
    import gc

    del encoder
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def exact_top_k_positions(
    product_matrix: np.ndarray, query_vector: np.ndarray, *, k: int
) -> list[int]:
    """Exhaustive cosine top-k over L2-normalized embeddings."""
    scores = product_matrix @ query_vector
    partition = np.argpartition(-scores, k - 1)[:k]
    ordered = partition[np.argsort(-scores[partition], kind="stable")]
    return [int(position) for position in ordered]


def paired_verdict(
    incumbent_values: np.ndarray, challenger_values: np.ndarray, *, seed: int
) -> dict[str, object]:
    """Paired bootstrap CI, sign-flip permutation p-value, and W/T/L counts."""
    deltas = challenger_values - incumbent_values
    rng = np.random.default_rng(seed)
    resampled = rng.choice(
        deltas, size=(BOOTSTRAP_RESAMPLES, len(deltas)), replace=True
    ).mean(axis=1)
    low, high = np.percentile(resampled, [2.5, 97.5])
    signs = rng.choice([-1.0, 1.0], size=(BOOTSTRAP_RESAMPLES, len(deltas)))
    null_means = (signs * np.abs(deltas)).mean(axis=1)
    observed = float(deltas.mean())
    p_value = float(
        (np.count_nonzero(np.abs(null_means) >= abs(observed)) + 1)
        / (BOOTSTRAP_RESAMPLES + 1)
    )
    return {
        "mean_delta": observed,
        "ci_95": [float(low), float(high)],
        "p_value": p_value,
        "wins": int(np.count_nonzero(deltas > 0)),
        "ties": int(np.count_nonzero(deltas == 0)),
        "losses": int(np.count_nonzero(deltas < 0)),
    }


def add_holm_adjustment(verdict_family: dict[str, dict[str, dict]]) -> None:
    """Add Holm-adjusted p-values across one family of contrasts, in place.

    Each family (todos los contrastes contra la titular de un nivel, o todos
    los directos de un nivel) se corrige por multiplicidad de forma conjunta:
    una afirmación de significación seleccionada entre muchos contrastes debe
    sobrevivir a la corrección, no solo a su p-valor nominal.
    """
    keyed = [
        (experiment, metric_name, metrics[metric_name]["p_value"])
        for experiment, metrics in verdict_family.items()
        for metric_name in metrics
    ]
    keyed.sort(key=lambda item: item[2])
    total = len(keyed)
    running_maximum = 0.0
    for position, (experiment, metric_name, p_value) in enumerate(keyed):
        adjusted = min(1.0, (total - position) * p_value)
        running_maximum = max(running_maximum, adjusted)
        verdict_family[experiment][metric_name]["p_value_holm"] = running_maximum


def main() -> None:
    """Rank every measured configuration on the expanded workload."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet", type=Path, default=PARQUET_PATH, help="ESCI examples parquet."
    )
    arguments = parser.parse_args()

    config = load_run_config()
    catalog = load_catalog()
    product_ids = catalog["product_id"].tolist()
    examples = load_public_judgments(arguments.parquet)
    judgments, queries, tier_queries = build_tiers(examples, set(product_ids))
    print(
        f"Consultas ampliadas: {len(queries)} "
        f"({', '.join(f'{name}={len(frame)}' for name, frame in tier_queries.items())})"
    )

    # Sin línea base BM25: con miles de consultas el índice puro-Python de la
    # sesión tardaría una hora, y la decisión aquí es denso contra denso; la
    # comparación léxica vive en scripts/run_experiments.py.
    rankings_by_experiment: dict[str, dict[str, list[str]]] = {}
    parameters_by_experiment: dict[str, dict[str, object]] = {}

    for name in sorted(EMBEDDING_CONFIGURATIONS):
        configuration = EMBEDDING_CONFIGURATIONS[name]
        if configuration.provider != "local":
            continue
        try:
            embedding_set = load_embedding_set(name)
        except FileNotFoundError:
            print(f"Omitida {name}: no hay embeddings persistidos.")
            continue
        product_matrix = embedding_set.matrix(SET_PRODUCTS)
        encoder = load_encoder(configuration.model_id)
        query_matrix = encode_texts(
            encoder,
            queries["query_text"].tolist(),
            prefix=configuration.query_prefix,
            normalize=configuration.normalize,
        )
        _release_encoder(encoder)
        rankings_by_experiment[name] = {
            query["query_id"]: [
                product_ids[position]
                for position in exact_top_k_positions(
                    product_matrix, query_matrix[row_position], k=config.top_k
                )
            ]
            for row_position, (_, query) in enumerate(queries.iterrows())
        }
        parameters_by_experiment[name] = {
            "kind": "dense_exact",
            "model_id": configuration.model_id,
            "dimension": configuration.dimension,
        }
        print(f"Codificada y evaluada {name}.")

    if INCUMBENT not in rankings_by_experiment:
        raise ValueError(
            f"Faltan los embeddings de la configuración titular {INCUMBENT!r}: "
            "la comparación pareada no tiene referencia. Genera los ficheros "
            "con `make embeddings`."
        )

    tiers_report: dict[str, object] = {}
    tables: list[pd.DataFrame] = []
    for tier_name, tier_frame in tier_queries.items():
        query_ids = tier_frame["query_id"].tolist()
        per_experiment_metrics: dict[str, list] = {}
        for experiment, rankings in rankings_by_experiment.items():
            per_experiment_metrics[experiment] = [
                evaluate_query(
                    query_id,
                    rankings[query_id],
                    judgments,
                    k=config.top_k,
                    recall_relevant_labels=config.recall_relevant_labels,
                    mrr_relevant_labels=config.mrr_relevant_labels,
                )
                for query_id in query_ids
            ]

        def metric_vector(
            experiment: str, attribute: str, *, metrics=per_experiment_metrics
        ) -> np.ndarray:
            return np.array(
                [getattr(metric, attribute) for metric in metrics[experiment]]
            )

        verdicts = {}
        for experiment in per_experiment_metrics:
            if experiment == INCUMBENT:
                continue
            verdicts[experiment] = {
                attribute: paired_verdict(
                    metric_vector(INCUMBENT, attribute),
                    metric_vector(experiment, attribute),
                    seed=config.random_seed,
                )
                for attribute in METRIC_ATTRIBUTES
            }
        add_holm_adjustment(verdicts)

        direct = {}
        for reference, challenger in DIRECT_PAIRS:
            if (
                reference not in per_experiment_metrics
                or challenger not in per_experiment_metrics
            ):
                continue
            direct[f"{challenger}_vs_{reference}"] = {
                attribute: paired_verdict(
                    metric_vector(reference, attribute),
                    metric_vector(challenger, attribute),
                    seed=config.random_seed,
                )
                for attribute in METRIC_ATTRIBUTES
            }
        add_holm_adjustment(direct)

        tier_table = pd.DataFrame(
            [
                {
                    "tier": tier_name,
                    "experiment": experiment,
                    "queries": len(query_ids),
                    **macro_average(metrics),
                }
                for experiment, metrics in per_experiment_metrics.items()
            ]
        )
        tables.append(tier_table)
        tiers_report[tier_name] = {
            "minimum_judged_in_catalog": TIERS[tier_name],
            "queries": len(query_ids),
            "aggregates": {
                experiment: macro_average(metrics)
                for experiment, metrics in per_experiment_metrics.items()
            },
            "paired_versus_incumbent": verdicts,
            "paired_direct": direct,
            # Métricas por consulta: permiten cualquier comparación pareada
            # a posteriori (no solo contra la configuración titular).
            "per_query": {
                experiment: [metric.as_record() for metric in metrics]
                for experiment, metrics in per_experiment_metrics.items()
            },
        }

    table = pd.concat(tables, ignore_index=True)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": {
            "dataset": "Shopping Queries Dataset (ESCI), locale es",
            "parquet_sha256": hashlib.sha256(
                arguments.parquet.read_bytes()
            ).hexdigest(),
            "url": PARQUET_URL,
        },
        "method": {
            "incumbent": INCUMBENT,
            "k": config.top_k,
            "recall_relevant_labels": list(config.recall_relevant_labels),
            "mrr_relevant_labels": list(config.mrr_relevant_labels),
            "selection": "E>=1 y E+S>=2 en catálogo; juicios restringidos al catálogo",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "random_seed": config.random_seed,
            "multiplicity": (
                "p_value_holm: corrección de Holm por familia de contrastes "
                "(todos los pareados contra la titular de un nivel; los "
                "directos de un nivel)"
            ),
        },
        "experiments": parameters_by_experiment,
        "tiers": tiers_report,
    }
    path = write_json_artifact(
        report, ARTIFACTS_DIRECTORY / "experimentos" / "validacion_ampliada.json"
    )
    table_path = ARTIFACTS_DIRECTORY / "experimentos" / "tabla_validacion_ampliada.csv"
    table.to_csv(table_path, index=False)
    print(table.to_string(index=False))
    print(f"Registro escrito en {path.relative_to(PROJECT_ROOT)}")
    print(f"Tabla escrita en {table_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
