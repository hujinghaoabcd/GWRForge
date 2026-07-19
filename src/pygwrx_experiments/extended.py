"""Parameter selection, ablation, benchmarking, and paired statistics."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
from pygwrx import GRGWR, LGGWR
from scipy.stats import wilcoxon

from .config import ExperimentConfig
from .metrics import coefficient_metrics, geometry_metrics, regime_metrics, regression_metrics
from .models import fit_grgwr
from .simulations import make_gr_scenario, make_lg_scenario


def run_gr_parameter_selection(config: ExperimentConfig) -> pd.DataFrame:
    """Run a reproducible spatial-CV parameter search on controlled R2 data."""
    data = make_gr_scenario("R2", side=config.grid_side, n_test=config.n_test, seed=config.random_seed + 700000)
    if config.profile == "pilot":
        regime_grid = (1, 2)
        bandwidth_grid = (12, 16)
        boundary_grid = (0.0, 1.0)
        spatial_grid = (0.25, 0.75)
        max_iter = 1
    else:
        regime_grid = (1, 2, 3, 4, 5, 6)
        bandwidth_grid = (20, 30, 40, 60, 80)
        boundary_grid = (0.0, 0.1, 0.5, 1.0, 2.0, 5.0)
        spatial_grid = (0.0, 0.25, 0.5, 0.75, 1.0)
        max_iter = config.gr_max_iter
    best, table = GRGWR.select_parameters(
        data.X_train,
        data.y_train,
        data.coords_train,
        n_regimes_grid=regime_grid,
        bandwidth_grid=bandwidth_grid,
        lambda_boundary_grid=boundary_grid,
        spatial_constraint_grid=spatial_grid,
        criterion="spatial_cv",
        cv_folds=config.cv_splits,
        max_iter=max_iter,
        n_neighbors=min(8, len(data.X_train) - 1),
        random_state=config.random_seed,
    )
    table = table.copy()
    table["selected"] = False
    selected = int(table["score"].astype(float).idxmin())
    table.loc[selected, "selected"] = True
    result = best.predict_result(data.X_test, data.coords_test)
    for key, value in regression_metrics(data.y_test, result.predictions).items():
        table.loc[selected, f"independent_{key}"] = value
    table.loc[selected, "selected_regimes_actual"] = best.n_regimes_actual_
    return table


def run_lg_ablation(config: ExperimentConfig) -> pd.DataFrame:
    """Run core LG-GWR ablations on L3 geography-plus-attribute geometry."""
    data = make_lg_scenario("L3", side=config.grid_side, n_test=config.n_test, seed=config.random_seed + 710000)
    bandwidth = max(12, min(len(data.X_train) - 1, len(data.X_train) // 4))
    rng = np.random.default_rng(config.random_seed)
    shuffled = data.attrs_train.copy()
    rng.shuffle(shuffled, axis=0)
    variants = [
        ("joint_coords_only", "joint", None, None, 2, "coordinate", 1, 0),
        ("joint_coords_attributes", "joint", data.attrs_train, data.attrs_test, 2, "coordinate", config.lg_restarts, 0),
        ("separable", "separable", data.attrs_train, data.attrs_test, 2, "coordinate", config.lg_restarts, 0),
        ("shuffled_attributes", "joint", shuffled, data.attrs_test, 2, "coordinate", config.lg_restarts, 0),
    ]
    for latent_dim in (1, 2, 3, 4):
        variants.append((f"latent_dim_{latent_dim}", "joint", data.attrs_train, data.attrs_test, latent_dim, "coordinate", 1, 0))
    rows: list[dict[str, object]] = []
    for name, geometry, attrs_train, attrs_test, latent_dim, initialization, restarts, updates in variants:
        started = perf_counter()
        bandwidth_spec = bandwidth if geometry == "joint" else (bandwidth, bandwidth)
        model = LGGWR(
            latent_dim=latent_dim,
            bandwidth=bandwidth_spec,
            adaptive=True,
            geometry=geometry,
            kernel="gaussian",
            max_iter=config.lg_max_iter,
            n_restarts=restarts,
            bandwidth_updates=updates,
            select_bandwidth=False,
            initialization=initialization,
            random_state=config.random_seed,
        ).fit(data.X_train, data.y_train, data.coords_train, attrs_train)
        result = model.predict_result(data.X_test, data.coords_test, attrs_test)
        row: dict[str, object] = {
            "variant": name,
            "geometry": geometry,
            "latent_dim": latent_dim,
            "fit_seconds": perf_counter() - started,
            "n_iter": model.n_iter_,
            "converged": model.converged_,
            "train_aicc": model.diagnostics_["aicc"],
        }
        row.update(regression_metrics(data.y_test, result.predictions))
        row.update(coefficient_metrics(data.beta_test, result.coefficients))
        if data.z_train is not None and model.latent_coords_.shape[1] >= 2:
            row.update(geometry_metrics(data.z_train, model.latent_coords_))
        rows.append(row)
    return pd.DataFrame(rows)


def run_gr_ablation(config: ExperimentConfig) -> pd.DataFrame:
    """Run core GR-GWR ablations on irregular connected scenario R3."""
    data = make_gr_scenario("R3", side=config.grid_side, n_test=config.n_test, seed=config.random_seed + 720000)
    bandwidth = max(12, min(len(data.X_train) - 1, len(data.X_train) // 4))
    variants = [
        ("full", config.gr_max_iter, 1.0, 0.5, True),
        ("no_icm", 0, 1.0, 0.5, True),
        ("no_boundary_penalty", config.gr_max_iter, 0.0, 0.5, True),
        ("no_connectivity", config.gr_max_iter, 1.0, 0.5, False),
        ("coefficients_only_gamma0", config.gr_max_iter, 1.0, 0.0, True),
        ("coordinates_only_gamma1", config.gr_max_iter, 1.0, 1.0, True),
    ]
    rows: list[dict[str, object]] = []
    for name, max_iter, boundary, gamma, connectivity in variants:
        output = fit_grgwr(
            data.X_train,
            data.y_train,
            data.coords_train,
            data.X_test,
            data.coords_test,
            n_regimes=3,
            bandwidth=bandwidth,
            max_iter=max_iter,
            seed=config.random_seed,
            lambda_boundary=boundary,
            spatial_constraint_weight=gamma,
            enforce_connectivity=connectivity,
        )
        row: dict[str, object] = {
            "variant": name,
            "fit_seconds": output.fit_seconds,
            "n_iter": output.model.n_iter_,
            "converged": output.model.converged_,
            "objective": output.model.diagnostics_["objective"],
            "n_boundaries": output.model.diagnostics_["n_boundaries"],
        }
        row.update(regression_metrics(data.y_test, output.predictions))
        row.update(coefficient_metrics(data.beta_test, output.coefficients))
        row.update(regime_metrics(data.regimes_train, output.model.regimes_, data.coords_train))
        rows.append(row)
    return pd.DataFrame(rows)


def run_runtime_benchmark(config: ExperimentConfig) -> pd.DataFrame:
    """Measure fit time over increasing n without claiming hardware portability."""
    sides = (5, 7, 9) if config.profile == "pilot" else (10, 15, 20)
    rows: list[dict[str, object]] = []
    for side in sides:
        lg_data = make_lg_scenario("L3", side=side, n_test=16, seed=config.random_seed + side)
        gr_data = make_gr_scenario("R3", side=side, n_test=16, seed=config.random_seed + side)
        bandwidth = max(10, min(side * side - 1, side * side // 4))
        started = perf_counter()
        LGGWR(
            latent_dim=2,
            bandwidth=bandwidth,
            adaptive=True,
            geometry="joint",
            max_iter=min(config.lg_max_iter, 10),
            select_bandwidth=False,
            bandwidth_updates=0,
            random_state=config.random_seed,
        ).fit(lg_data.X_train, lg_data.y_train, lg_data.coords_train, lg_data.attrs_train)
        rows.append({"n": side * side, "method": "LG-GWR Joint", "fit_seconds": perf_counter() - started})
        output = fit_grgwr(
            gr_data.X_train,
            gr_data.y_train,
            gr_data.coords_train,
            gr_data.X_test,
            gr_data.coords_test,
            n_regimes=3,
            bandwidth=bandwidth,
            max_iter=min(config.gr_max_iter, 2),
            seed=config.random_seed,
        )
        rows.append({"n": side * side, "method": "GR-GWR", "fit_seconds": output.fit_seconds})
    return pd.DataFrame(rows)


def paired_statistics(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Compare each method against GWR using paired errors and Holm correction."""
    if frame.empty or "GWR" not in set(frame["method"]):
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for candidate in sorted(set(frame["method"]) - {"GWR"}):
        for group, subset in frame.groupby(group_column):
            index_columns = [column for column in ("repetition", "repeat", "fold") if column in subset.columns]
            pivot = subset.pivot_table(index=index_columns, columns="method", values="rmse", aggfunc="mean")
            if "GWR" not in pivot or candidate not in pivot:
                continue
            paired = pivot[["GWR", candidate]].dropna()
            if len(paired) < 2:
                continue
            difference = paired[candidate] - paired["GWR"]
            try:
                statistic, p_value = wilcoxon(difference)
            except ValueError:
                statistic, p_value = 0.0, 1.0
            rows.append({
                group_column: group,
                "candidate": candidate,
                "n_pairs": len(paired),
                "mean_rmse_difference_vs_gwr": float(difference.mean()),
                "wilcoxon_statistic": float(statistic),
                "p_value": float(p_value),
            })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values("p_value").reset_index(drop=True)
    m = len(result)
    adjusted = np.maximum.accumulate([(m - index) * value for index, value in enumerate(result["p_value"])])
    result["holm_p_value"] = np.minimum(adjusted, 1.0)
    return result
