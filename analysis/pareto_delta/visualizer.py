"""
analysis/pareto_delta/visualizer.py — Optional matplotlib/plotly charts for
trajectory analysis and Pareto rankings.

All functions return figure objects; call .show() or .savefig() as needed.
Requires the [viz] optional dependency group:
    pip install 'autoscientist-etft[viz]'
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.figure
    import plotly.graph_objects


def plot_fitness_trajectory(
    step_indices: list[int],
    fitness_scores: list[float],
    title: str = "Evolutionary Fitness Trajectory",
) -> matplotlib.figure.Figure:
    """Line chart of fitness score across trajectory steps."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Install matplotlib: pip install 'autoscientist-etft[viz]'") from exc

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(step_indices, fitness_scores, marker="o", linewidth=2, color="#2563eb")
    ax.set_xlabel("Trajectory Step")
    ax.set_ylabel("Fitness ℱ(a)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_pareto_bar(
    component_names: list[str],
    contributions: list[float],
    pareto_threshold: float = 0.80,
    title: str = "Retrospective 80/20 Δ Analysis",
) -> matplotlib.figure.Figure:
    """Pareto bar chart with a cumulative contribution line."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Install matplotlib: pip install 'autoscientist-etft[viz]'") from exc

    total = sum(contributions) or 1.0
    fractions = [c / total for c in contributions]
    cumulative = list(__import__("itertools").accumulate(fractions))

    fig, ax1 = plt.subplots(figsize=(max(6, len(component_names) * 0.8), 5))
    x = range(len(component_names))

    ax1.bar(x, fractions, color="#dbeafe", edgecolor="#2563eb")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(component_names, rotation=30, ha="right")
    ax1.set_ylabel("Fractional Contribution")
    ax1.set_title(title)

    ax2 = ax1.twinx()
    ax2.plot(list(x), cumulative, "r-o", linewidth=2, label="Cumulative")
    ax2.axhline(pareto_threshold, color="gray", linestyle="--", label=f"{pareto_threshold*100:.0f}% threshold")
    ax2.set_ylabel("Cumulative Fraction")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="lower right")

    fig.tight_layout()
    return fig


def plot_pareto_interactive(
    component_names: list[str],
    contributions: list[float],
    pareto_threshold: float = 0.80,
    title: str = "Retrospective 80/20 Δ Analysis (Interactive)",
) -> plotly.graph_objects.Figure:
    """Interactive Plotly Pareto chart."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("Install plotly: pip install 'autoscientist-etft[viz]'") from exc

    import itertools

    total = sum(contributions) or 1.0
    fractions = [c / total for c in contributions]
    cumulative = list(itertools.accumulate(fractions))

    fig = go.Figure()
    fig.add_trace(go.Bar(x=component_names, y=fractions, name="Contribution", marker_color="#2563eb"))
    fig.add_trace(
        go.Scatter(
            x=component_names, y=cumulative, name="Cumulative", mode="lines+markers",
            line={"color": "red", "width": 2}, yaxis="y2",
        )
    )
    fig.add_hline(y=pareto_threshold, line_dash="dash", line_color="gray", yref="y2")
    fig.update_layout(
        title=title,
        yaxis={"title": "Fractional Contribution"},
        yaxis2={"title": "Cumulative Fraction", "overlaying": "y", "side": "right", "range": [0, 1.05]},
        legend={"x": 0.7, "y": 0.1},
    )
    return fig
