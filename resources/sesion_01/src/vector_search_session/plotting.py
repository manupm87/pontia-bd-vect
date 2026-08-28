"""Professional Plotly-only visualizations shared by the session notebooks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike

from .evaluation import EvaluationReport
from .retrieval import LatencySummary, SearchResult

PRIMARY_COLOR = "#2563EB"
SECONDARY_COLOR = "#14B8A6"
ACCENT_COLOR = "#F59E0B"
NEGATIVE_COLOR = "#DC2626"
TEXT_COLOR = "#172033"
MUTED_TEXT_COLOR = "#64748B"
GRID_COLOR = "#E2E8F0"
BACKGROUND_COLOR = "#FFFFFF"
COLOR_SEQUENCE = (
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    ACCENT_COLOR,
    "#8B5CF6",
    "#EC4899",
    "#0EA5E9",
    "#84CC16",
    "#F97316",
)


def apply_professional_layout(
    figure: go.Figure,
    *,
    title: str,
    subtitle: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    height: int = 520,
) -> go.Figure:
    """Apply one consistent, accessible visual language to a Plotly figure."""

    if not isinstance(title, str) or not title.strip():
        raise ValueError("title debe ser un string no vacío.")
    if height < 300:
        raise ValueError("height debe ser al menos 300 píxeles.")
    title_text = f"<b>{title}</b>"
    if subtitle:
        title_text += (
            f"<br><span style='font-size:13px;color:{MUTED_TEXT_COLOR}'>"
            f"{subtitle}</span>"
        )

    figure.update_layout(
        template="plotly_white",
        title={"text": title_text, "x": 0.02, "xanchor": "left", "y": 0.97},
        height=height,
        margin={"l": 72, "r": 32, "t": 96, "b": 64},
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        font={"family": "Arial, sans-serif", "size": 13, "color": TEXT_COLOR},
        colorway=list(COLOR_SEQUENCE),
        hoverlabel={"bgcolor": "white", "font_size": 13, "font_family": "Arial"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
        },
    )
    figure.update_xaxes(
        title=xaxis_title,
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
        automargin=True,
    )
    figure.update_yaxes(
        title=yaxis_title,
        showgrid=False,
        zeroline=False,
        automargin=True,
    )
    return figure


def plot_ranked_results(
    results: Sequence[SearchResult],
    *,
    title: str,
    document_labels: Mapping[str, str] | None = None,
    score_label: str = "Puntuación",
    subtitle: str | None = None,
) -> go.Figure:
    """Create a horizontal result chart with rank and score in the hover."""

    if not results:
        raise ValueError("results debe contener al menos un resultado.")
    reversed_results = tuple(reversed(results))
    labels = [
        (document_labels or {}).get(result.document_id, result.document_id)
        for result in reversed_results
    ]
    custom_data = np.asarray(
        [[result.rank, result.document_id] for result in reversed_results],
        dtype=object,
    )
    figure = go.Figure(
        go.Bar(
            x=[result.score for result in reversed_results],
            y=labels,
            orientation="h",
            marker={"color": PRIMARY_COLOR, "line": {"width": 0}},
            customdata=custom_data,
            hovertemplate=(
                "<b>%{y}</b><br>Rank: %{customdata[0]}"
                f"<br>{score_label}: %{{x:.4f}}"
                "<br>ID: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    return apply_professional_layout(
        figure,
        title=title,
        subtitle=subtitle,
        xaxis_title=score_label,
        yaxis_title=None,
        height=max(420, 90 + 48 * len(results)),
    )


def _summary_from_report_or_mapping(
    value: EvaluationReport | Mapping[str, float],
) -> Mapping[str, float]:
    return value.summary if isinstance(value, EvaluationReport) else value


def plot_metric_comparison(
    systems: Mapping[str, EvaluationReport | Mapping[str, float]],
    *,
    title: str = "Calidad de recuperación por sistema",
    subtitle: str | None = None,
) -> go.Figure:
    """Compare retrieval metrics on a common 0-1 scale."""

    if not systems:
        raise ValueError("systems no puede estar vacío.")
    summaries = {
        system_name: dict(_summary_from_report_or_mapping(value))
        for system_name, value in systems.items()
    }
    metric_names = list(next(iter(summaries.values())))
    if not metric_names:
        raise ValueError("Cada sistema debe aportar al menos una métrica.")
    expected_metrics = set(metric_names)

    figure = go.Figure()
    for system_name, summary in summaries.items():
        if set(summary) != expected_metrics:
            raise ValueError(
                "Todos los sistemas deben contener exactamente las mismas métricas."
            )
        metric_values = [float(summary[metric_name]) for metric_name in metric_names]
        if any(not np.isfinite(value) for value in metric_values):
            raise ValueError("Las métricas deben ser finitas.")
        figure.add_trace(
            go.Bar(
                name=system_name,
                x=metric_names,
                y=metric_values,
                text=[f"{value:.3f}" for value in metric_values],
                textposition="outside",
                cliponaxis=False,
                hovertemplate=(
                    f"<b>{system_name}</b><br>%{{x}}: %{{y:.4f}}<extra></extra>"
                ),
            )
        )
    figure.update_layout(barmode="group")
    figure.update_yaxes(range=[0, 1.08], tickformat=".0%")
    return apply_professional_layout(
        figure,
        title=title,
        subtitle=subtitle,
        xaxis_title="Métrica",
        yaxis_title="Resultado",
        height=520,
    )


def plot_latency_comparison(
    systems: Mapping[str, LatencySummary],
    *,
    title: str = "Latencia de búsqueda",
    subtitle: str = "Mediana y percentil 95; menor es mejor",
) -> go.Figure:
    """Compare p50 and p95 latency without hiding tail behavior."""

    if not systems:
        raise ValueError("systems no puede estar vacío.")
    system_names = list(systems)
    figure = go.Figure()
    for label, attribute, color in (
        ("p50", "p50_ms", PRIMARY_COLOR),
        ("p95", "p95_ms", ACCENT_COLOR),
    ):
        values = [float(getattr(systems[name], attribute)) for name in system_names]
        figure.add_trace(
            go.Bar(
                name=label,
                x=system_names,
                y=values,
                marker_color=color,
                text=[f"{value:.2f} ms" for value in values],
                textposition="outside",
                cliponaxis=False,
                hovertemplate=(
                    f"<b>%{{x}}</b><br>{label}: %{{y:.3f}} ms<extra></extra>"
                ),
            )
        )
    figure.update_layout(barmode="group")
    return apply_professional_layout(
        figure,
        title=title,
        subtitle=subtitle,
        xaxis_title="Sistema",
        yaxis_title="Latencia (ms)",
        height=500,
    )


def plot_similarity_heatmap(
    similarities: ArrayLike,
    *,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    title: str = "Interacción entre vectores",
    subtitle: str | None = None,
) -> go.Figure:
    """Display token-token or query-document similarities as a heatmap."""

    matrix = np.asarray(similarities, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("similarities debe ser una matriz bidimensional no vacía.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("similarities contiene NaN o infinito.")
    if len(row_labels) != matrix.shape[0]:
        raise ValueError("row_labels no coincide con el número de filas.")
    if len(column_labels) != matrix.shape[1]:
        raise ValueError("column_labels no coincide con el número de columnas.")

    maximum_absolute_value = max(float(np.max(np.abs(matrix))), 1e-12)
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=list(column_labels),
            y=list(row_labels),
            zmin=-maximum_absolute_value,
            zmax=maximum_absolute_value,
            zmid=0,
            colorscale=[
                [0.0, "#B91C1C"],
                [0.5, "#F8FAFC"],
                [1.0, PRIMARY_COLOR],
            ],
            colorbar={"title": "Similitud"},
            text=np.round(matrix, 3),
            texttemplate="%{text}",
            hovertemplate="Fila: %{y}<br>Columna: %{x}<br>Score: %{z:.4f}<extra></extra>",
        )
    )
    return apply_professional_layout(
        figure,
        title=title,
        subtitle=subtitle,
        xaxis_title="Vector del documento",
        yaxis_title="Vector de la consulta",
        height=max(460, 180 + 34 * matrix.shape[0]),
    )


def plot_embedding_projection(
    coordinates: ArrayLike,
    labels: Sequence[str],
    *,
    groups: Sequence[str] | None = None,
    title: str = "Proyección bidimensional de embeddings",
    subtitle: str = "La proyección ayuda a explorar; no sustituye la evaluación",
) -> go.Figure:
    """Plot precomputed 2D coordinates, optionally colored by group."""

    coordinate_matrix = np.asarray(coordinates, dtype=np.float64)
    if coordinate_matrix.ndim != 2 or coordinate_matrix.shape[1] != 2:
        raise ValueError("coordinates debe tener forma (n_elementos, 2).")
    if coordinate_matrix.shape[0] == 0 or not np.all(np.isfinite(coordinate_matrix)):
        raise ValueError("coordinates debe ser no vacío y contener valores finitos.")
    if len(labels) != coordinate_matrix.shape[0]:
        raise ValueError("labels no coincide con el número de coordenadas.")
    normalized_groups = (
        tuple(groups) if groups is not None else ("Elementos",) * len(labels)
    )
    if len(normalized_groups) != len(labels):
        raise ValueError("groups no coincide con el número de coordenadas.")

    figure = go.Figure()
    group_order = tuple(dict.fromkeys(normalized_groups))
    for group_index, group_name in enumerate(group_order):
        point_indices = [
            index
            for index, current_group in enumerate(normalized_groups)
            if current_group == group_name
        ]
        figure.add_trace(
            go.Scatter(
                x=coordinate_matrix[point_indices, 0],
                y=coordinate_matrix[point_indices, 1],
                mode="markers",
                name=group_name,
                text=[labels[index] for index in point_indices],
                marker={
                    "size": 11,
                    "color": COLOR_SEQUENCE[group_index % len(COLOR_SEQUENCE)],
                    "line": {"color": "white", "width": 1},
                    "opacity": 0.88,
                },
                hovertemplate="<b>%{text}</b><br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
            )
        )
    return apply_professional_layout(
        figure,
        title=title,
        subtitle=subtitle,
        xaxis_title="Componente 1",
        yaxis_title="Componente 2",
        height=600,
    )
