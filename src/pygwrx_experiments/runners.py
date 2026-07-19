"""High-level execution functions for simulations and real-data validation."""

from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from pygwrx.io import load_dataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .config import ExperimentConfig
from .metrics import coefficient_metrics, geometry_metrics, regime_metrics, regression_metrics
from .models import ModelOutput, fit_grgwr, fit_gwr, fit_lggwr, fit_ols, fit_optional_mgwr, fit_optional_sgwr
from .simulations import make_gr_scenario, make_lg_scenario


def _record_output(output: ModelOutput, y_true: np.ndarray, beta_true: np.ndarray | None) -> dict[str, float | str]:
    row: dict[str, float | str] = {"method": output.method, "fit_seconds": output.fit_seconds}
    row.update(regression_metrics(y_true, output.predictions))
    if beta_true is not None and np.isfinite(output.coefficients).all():
        row.update(coefficient_metrics(beta_true, output.coefficients))
    return row


def run_lg_simulations(config: ExperimentConfig) -> pd.DataFrame:
    """Run L1--L6 with deterministic Monte Carlo repetitions."""
    rows: list[dict[str, object]] = []
    for scenario_index in range(1, 7):
        scenario = f"L{scenario_index}"
        for repetition in range(config.simulation_repetitions):
            seed = config.random_seed + scenario_index * 10000 + repetition
            data = make_lg_scenario(scenario, side=config.grid_side, n_test=config.n_test, seed=seed)
            bandwidth = max(12, min(len(data.X_train) - 1, len(data.X_train) // 4))
            methods: list[ModelOutput] = [
                fit_ols(data.X_train, data.y_train, data.X_test),
                fit_gwr(data.X_train, data.y_train, data.coords_train, data.X_test, data.coords_test, bandwidth=bandwidth),
            ]
            for geometry in ("joint", "separable"):
                methods.append(
                    fit_lggwr(
                        data.X_train,
                        data.y_train,
                        data.coords_train,
                        data.attrs_train,
                        data.X_test,
                        data.coords_test,
                        data.attrs_test,
                        geometry=geometry,
                        bandwidth=bandwidth,
                        max_iter=config.lg_max_iter,
                        n_restarts=config.lg_restarts,
                        seed=seed,
                    )
                )
            if config.run_mgwr:
                methods.append(fit_optional_mgwr(data.X_train, data.y_train, data.coords_train, data.X_test, data.coords_test))
            if config.run_sgwr:
                methods.append(fit_optional_sgwr(data.X_train, data.y_train, data.coords_train, data.X_test, data.coords_test))
            for output in methods:
                row = _record_output(output, data.y_test, data.beta_test)
                row.update({"scenario": scenario, "repetition": repetition, "seed": seed})
                if output.method == "LG-GWR Joint" and data.z_train is not None:
                    row.update(geometry_metrics(data.z_train, output.model.latent_coords_))
                    for index, value in enumerate(output.model.metric_contributions_):
                        row[f"metric_contribution_{index}"] = float(value)
                    row["converged"] = bool(output.model.converged_)
                    row["n_iter"] = int(output.model.n_iter_)
                rows.append(row)
    return pd.DataFrame(rows)


def _scenario_regime_count(scenario: str) -> int:
    return {"R1": 1, "R2": 2, "R3": 3, "R4": 3, "R5": 1, "R6": 2}[scenario]


def run_gr_simulations(config: ExperimentConfig) -> pd.DataFrame:
    """Run R1--R6 with deterministic Monte Carlo repetitions."""
    rows: list[dict[str, object]] = []
    for scenario_index in range(1, 7):
        scenario = f"R{scenario_index}"
        for repetition in range(config.simulation_repetitions):
            seed = config.random_seed + 100000 + scenario_index * 10000 + repetition
            data = make_gr_scenario(scenario, side=config.grid_side, n_test=config.n_test, seed=seed)
            bandwidth = max(12, min(len(data.X_train) - 1, len(data.X_train) // 4))
            methods = [
                fit_ols(data.X_train, data.y_train, data.X_test),
                fit_gwr(data.X_train, data.y_train, data.coords_train, data.X_test, data.coords_test, bandwidth=bandwidth),
                fit_grgwr(
                    data.X_train,
                    data.y_train,
                    data.coords_train,
                    data.X_test,
                    data.coords_test,
                    n_regimes=_scenario_regime_count(scenario),
                    bandwidth=bandwidth,
                    max_iter=config.gr_max_iter,
                    seed=seed,
                ),
            ]
            for output in methods:
                row = _record_output(output, data.y_test, data.beta_test)
                row.update({"scenario": scenario, "repetition": repetition, "seed": seed})
                if output.method == "GR-GWR":
                    row.update(regime_metrics(data.regimes_train, output.model.regimes_, data.coords_train, neighbors=min(8, len(data.X_train) - 1)))
                    row["n_iter"] = int(output.model.n_iter_)
                    row["converged"] = bool(output.model.converged_)
                    row["objective"] = float(output.model.diagnostics_["objective"])
                rows.append(row)
    return pd.DataFrame(rows)


def spatial_blocks(coords: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    """Assign observations to approximately compact spatial blocks."""
    return KMeans(n_clusters=n_splits, random_state=seed, n_init=20).fit_predict(coords)


def _housing_sample(
    X: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a reproducible spatially stratified Housing sample."""
    if len(X) <= size:
        return X, y, coords, np.arange(len(X))
    strata = KMeans(n_clusters=20, random_state=seed, n_init=20).fit_predict(coords)
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for label in np.unique(strata):
        candidates = np.flatnonzero(strata == label)
        quota = max(1, round(size * len(candidates) / len(X)))
        chosen.extend(rng.choice(candidates, size=min(quota, len(candidates)), replace=False).tolist())
    if len(chosen) > size:
        chosen = rng.choice(np.asarray(chosen), size=size, replace=False).tolist()
    elif len(chosen) < size:
        remaining = np.setdiff1d(np.arange(len(X)), np.asarray(chosen), assume_unique=False)
        chosen.extend(rng.choice(remaining, size=size - len(chosen), replace=False).tolist())
    index = np.sort(np.asarray(chosen, dtype=int))
    return X[index], y[index], coords[index], index


def run_real_data(config: ExperimentConfig, output_dir: str | Path) -> pd.DataFrame:
    """Run repeated spatial-block validation on bundled pyGWRx datasets."""
    rows: list[dict[str, object]] = []
    sample_dir = Path(output_dir) / "sample_indices"
    sample_dir.mkdir(parents=True, exist_ok=True)
    datasets = sorted(set(config.real_datasets_lg) | set(config.real_datasets_gr))
    for dataset_name in datasets:
        bundle = load_dataset(dataset_name, return_type="dict")
        X = np.asarray(bundle["data"], dtype=float)
        y = np.asarray(bundle["target"], dtype=float)
        coords = np.asarray(bundle["coords"], dtype=float)
        if dataset_name == "housing":
            X, y, coords, sample_index = _housing_sample(X, y, coords, config.housing_sample_size, config.random_seed)
            pd.DataFrame({"sample_index": sample_index}).to_csv(
                sample_dir / f"housing_n{len(sample_index)}_seed{config.random_seed}.csv", index=False
            )
        for repeat in range(config.cv_repeats):
            split_seed = config.random_seed + repeat
            labels = spatial_blocks(coords, config.cv_splits, split_seed)
            for fold in range(config.cv_splits):
                train = labels != fold
                test = ~train
                x_scaler = StandardScaler().fit(X[train])
                X_train = x_scaler.transform(X[train])
                X_test = x_scaler.transform(X[test])
                y_mean = float(np.mean(y[train]))
                y_scale = float(np.std(y[train])) or 1.0
                y_train = (y[train] - y_mean) / y_scale
                y_test = (y[test] - y_mean) / y_scale
                coords_train, coords_test = coords[train], coords[test]
                bandwidth = max(X_train.shape[1] + 4, min(len(X_train) - 1, max(20, len(X_train) // 5)))
                methods: list[ModelOutput] = [
                    fit_ols(X_train, y_train, X_test),
                    fit_gwr(X_train, y_train, coords_train, X_test, coords_test, bandwidth=bandwidth),
                ]
                if dataset_name in config.real_datasets_lg:
                    methods.extend([
                        fit_lggwr(
                            X_train,
                            y_train,
                            coords_train,
                            X_train,
                            X_test,
                            coords_test,
                            X_test,
                            geometry=geometry,
                            bandwidth=bandwidth,
                            max_iter=config.lg_max_iter,
                            n_restarts=config.lg_restarts,
                            seed=split_seed + fold,
                        )
                        for geometry in ("joint", "separable")
                    ])
                if dataset_name in config.real_datasets_gr:
                    candidate_regimes = 2 if len(X_train) < 100 else 3
                    methods.append(
                        fit_grgwr(
                            X_train,
                            y_train,
                            coords_train,
                            X_test,
                            coords_test,
                            n_regimes=candidate_regimes,
                            bandwidth=bandwidth,
                            max_iter=config.gr_max_iter,
                            seed=split_seed + fold,
                        )
                    )
                for output in methods:
                    row = _record_output(output, y_test, beta_true=None)
                    row.update({
                        "dataset": dataset_name,
                        "repeat": repeat,
                        "fold": fold,
                        "n_train": int(train.sum()),
                        "n_test": int(test.sum()),
                        "seed": split_seed,
                    })
                    if hasattr(output.model, "diagnostics_") and output.model.diagnostics_:
                        row["train_aicc"] = float(output.model.diagnostics_.get("aicc", np.nan))
                        row["train_enp"] = float(output.model.diagnostics_.get("enp", np.nan))
                    rows.append(row)
    return pd.DataFrame(rows)


def safe_run(callable_object, *args, **kwargs):
    """Execute an experiment and return either its result or a failure table."""
    try:
        return callable_object(*args, **kwargs)
    except Exception as error:  # pragma: no cover - operational guard
        return pd.DataFrame([{
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }])
