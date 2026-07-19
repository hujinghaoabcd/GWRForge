"""Generate a concise Markdown report from machine-readable result tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No results were produced._"
    return frame.to_markdown(index=False, floatfmt=".4f")


def generate_report(root: str | Path, profile: str) -> Path:
    """Create a result report without inventing values absent from CSV files."""
    root_path = Path(root)
    correctness_path = root_path / "results/correctness/correctness.csv"
    lg_path = root_path / "results/lg_simulation/lg_simulation.csv"
    gr_path = root_path / "results/gr_simulation/gr_simulation.csv"
    real_path = root_path / "results/real_data/spatial_cv.csv"
    parameter_path = root_path / "results/gr_simulation/parameter_selection.csv"
    lg_ablation_path = root_path / "results/lg_simulation/ablation.csv"
    gr_ablation_path = root_path / "results/gr_simulation/ablation.csv"
    benchmark_path = root_path / "results/logs/runtime_benchmark.csv"

    sections = [
        "# pyGWRx LG-GWR and GR-GWR Experiment Report",
        "",
        f"Execution profile: **{profile}**.",
        "",
        "This report is generated only from the CSV outputs in `results/`. The pilot profile validates the pipeline; it is not a substitute for the 100-repetition and repeated spatial-CV full profile.",
        "",
        "## 1. Numerical correctness",
    ]
    if correctness_path.exists():
        sections.extend(["", _markdown_table(pd.read_csv(correctness_path))])
    else:
        sections.extend(["", "_Correctness checks have not been executed._"])

    sections.extend(["", "## 2. LG-GWR simulations"])
    if lg_path.exists():
        lg = pd.read_csv(lg_path)
        summary = (
            lg.groupby(["scenario", "method"], as_index=False)
            .agg(rmse=("rmse", "mean"), coef_rmse=("coef_rmse", "mean"), fit_seconds=("fit_seconds", "mean"))
            .sort_values(["scenario", "rmse"])
        )
        sections.extend(["", _markdown_table(summary)])
    else:
        sections.extend(["", "_LG-GWR simulations have not been executed._"])

    sections.extend(["", "## 3. GR-GWR simulations"])
    if gr_path.exists():
        gr = pd.read_csv(gr_path)
        aggregations = {
            "rmse": ("rmse", "mean"),
            "coef_rmse": ("coef_rmse", "mean"),
            "fit_seconds": ("fit_seconds", "mean"),
        }
        if "ari" in gr.columns:
            aggregations["ari"] = ("ari", "mean")
        if "boundary_f1" in gr.columns:
            aggregations["boundary_f1"] = ("boundary_f1", "mean")
        summary = gr.groupby(["scenario", "method"], as_index=False).agg(**aggregations).sort_values(["scenario", "rmse"])
        sections.extend(["", _markdown_table(summary)])
    else:
        sections.extend(["", "_GR-GWR simulations have not been executed._"])

    sections.extend(["", "## 4. Spatial out-of-sample validation"])
    if real_path.exists():
        real = pd.read_csv(real_path)
        summary = (
            real.groupby(["dataset", "method"], as_index=False)
            .agg(
                mae=("mae", "mean"),
                rmse=("rmse", "mean"),
                test_r2=("r2", "mean"),
                fold_sd=("rmse", "std"),
                fit_seconds=("fit_seconds", "mean"),
            )
            .sort_values(["dataset", "rmse"])
        )
        sections.extend(["", _markdown_table(summary)])
    else:
        sections.extend(["", "_Real-data spatial validation has not been executed._"])

    sections.extend(["", "## 5. GR-GWR spatial-CV parameter selection"])
    if parameter_path.exists():
        parameter = pd.read_csv(parameter_path)
        columns = [
            column
            for column in (
                "n_regimes",
                "bandwidth",
                "lambda_boundary",
                "spatial_constraint_weight",
                "score",
                "selected",
                "independent_rmse",
                "selected_regimes_actual",
            )
            if column in parameter.columns
        ]
        sections.extend(["", _markdown_table(parameter.sort_values("score")[columns].head(10))])
    else:
        sections.extend(["", "_Parameter selection has not been executed._"])

    sections.extend(["", "## 6. Ablation studies"])
    if lg_ablation_path.exists():
        lg_ablation = pd.read_csv(lg_ablation_path)
        columns = [column for column in ("variant", "latent_dim", "rmse", "coef_rmse", "distance_correlation", "knn_jaccard", "n_iter", "converged") if column in lg_ablation.columns]
        sections.extend(["", "### LG-GWR", "", _markdown_table(lg_ablation[columns].sort_values("rmse"))])
    if gr_ablation_path.exists():
        gr_ablation = pd.read_csv(gr_ablation_path)
        columns = [column for column in ("variant", "rmse", "coef_rmse", "ari", "boundary_f1", "n_boundaries", "n_iter", "converged") if column in gr_ablation.columns]
        sections.extend(["", "### GR-GWR", "", _markdown_table(gr_ablation[columns].sort_values("rmse"))])
    if not lg_ablation_path.exists() and not gr_ablation_path.exists():
        sections.extend(["", "_Ablation studies have not been executed._"])

    sections.extend(["", "## 7. Pilot runtime benchmark"])
    if benchmark_path.exists():
        sections.extend(["", _markdown_table(pd.read_csv(benchmark_path))])
    else:
        sections.extend(["", "_Runtime benchmark has not been executed._"])

    sections.extend([
        "",
        "## 8. Interpretation boundary",
        "",
        "- A pilot advantage is a software-validation observation, not a paper conclusion.",
        "- The full profile must be executed before significance tests or substantive claims.",
        "- LG-GWR geometry recovery is assessed through pairwise distances and neighbors, not raw rotation-dependent matrix entries.",
        "- GR-GWR conditional AICc is reported only as a fitted-label diagnostic; spatial-CV is the primary selection evidence.",
        "- Failures and non-convergence must remain in the final result tables rather than being silently removed.",
    ])
    output = root_path / f"reports/RESULTS_{profile.upper()}.md"
    output.write_text("\n".join(sections) + "\n", encoding="utf-8")
    return output
