"""Paper-oriented grayscale summary figures generated from result CSV files."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def grouped_rmse(frame: pd.DataFrame, group: str, output: Path, title: str) -> None:
    """Plot mean RMSE by scenario/dataset and method."""
    summary = frame.groupby([group, "method"], as_index=False)["rmse"].mean()
    groups = list(summary[group].drop_duplicates())
    methods = list(summary["method"].drop_duplicates())
    x = np.arange(len(groups), dtype=float)
    width = 0.8 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for index, method in enumerate(methods):
        values = []
        for item in groups:
            selected = summary[(summary[group] == item) & (summary["method"] == method)]
            values.append(float(selected["rmse"].iloc[0]) if len(selected) else np.nan)
        ax.bar(
            x + (index - (len(methods) - 1) / 2) * width,
            values,
            width,
            label=method,
            edgecolor="black",
            linewidth=0.6,
        )
    ax.set_xticks(x, groups)
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, output)


def correctness_figure(frame: pd.DataFrame, output: Path) -> None:
    """Plot numerical degeneration errors on a logarithmic scale."""
    labels = ["LG→GWR", "GR(1)→GWR"]
    coef = frame["max_coef_diff"].to_numpy()
    fitted = frame["max_fitted_diff"].to_numpy()
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(x - 0.18, coef, 0.36, label="Max coefficient difference", edgecolor="black")
    ax.bar(x + 0.18, fitted, 0.36, label="Max fitted-value difference", edgecolor="black")
    ax.axhline(1.0e-6, linestyle="--", linewidth=1.0, label="Acceptance threshold")
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Absolute difference")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, output)


def generate_all(root: str | Path) -> list[Path]:
    """Generate every figure for which a source CSV is available."""
    root_path = Path(root)
    figure_dir = root_path / "results/figures"
    outputs: list[Path] = []
    sources = [
        (root_path / "results/lg_simulation/lg_simulation.csv", "scenario", figure_dir / "lg_simulation_rmse.png", "LG-GWR simulation prediction error"),
        (root_path / "results/gr_simulation/gr_simulation.csv", "scenario", figure_dir / "gr_simulation_rmse.png", "GR-GWR simulation prediction error"),
        (root_path / "results/real_data/spatial_cv.csv", "dataset", figure_dir / "real_data_spatial_cv_rmse.png", "Spatial-block prediction error"),
    ]
    for source, group, output, title in sources:
        if source.exists():
            grouped_rmse(pd.read_csv(source), group, output, title)
            outputs.append(output)
    correctness = root_path / "results/correctness/correctness.csv"
    if correctness.exists():
        output = figure_dir / "correctness_differences.png"
        correctness_figure(pd.read_csv(correctness), output)
        outputs.append(output)
    return outputs
