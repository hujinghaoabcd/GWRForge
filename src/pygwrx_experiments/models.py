"""Model adapters that use the public pyGWRx API and consistent result shapes."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from pygwrx import GRGWR, GWR, LGGWR, MGWR, SGWR


@dataclass(frozen=True)
class ModelOutput:
    """Normalized fit and prediction output for one model execution."""

    method: str
    predictions: np.ndarray
    coefficients: np.ndarray
    intercepts: np.ndarray
    fit_seconds: float
    model: Any


def fit_ols(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> ModelOutput:
    """Fit an ordinary least-squares reference model."""
    started = perf_counter()
    design = np.column_stack([np.ones(len(X_train)), X_train])
    beta = np.linalg.lstsq(design, y_train, rcond=None)[0]
    predictions = np.column_stack([np.ones(len(X_test)), X_test]) @ beta
    elapsed = perf_counter() - started
    return ModelOutput(
        method="OLS",
        predictions=predictions,
        coefficients=np.tile(beta[1:], (len(X_test), 1)),
        intercepts=np.full(len(X_test), beta[0]),
        fit_seconds=elapsed,
        model={"beta": beta},
    )


def fit_gwr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    coords_train: np.ndarray,
    X_test: np.ndarray,
    coords_test: np.ndarray,
    *,
    bandwidth: int,
) -> ModelOutput:
    """Fit standard adaptive bisquare GWR from pyGWRx."""
    started = perf_counter()
    model = GWR(kernel="bisquare", bandwidth=bandwidth, adaptive=True, fit_intercept=True).fit(
        X_train, y_train, coords_train, compute_inference=False
    )
    result = model.predict_result(X_test, coords_test)
    return ModelOutput(
        method="GWR",
        predictions=result.predictions,
        coefficients=result.coef,
        intercepts=result.intercept,
        fit_seconds=perf_counter() - started,
        model=model,
    )


def fit_lggwr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    coords_train: np.ndarray,
    attrs_train: np.ndarray,
    X_test: np.ndarray,
    coords_test: np.ndarray,
    attrs_test: np.ndarray,
    *,
    geometry: str,
    bandwidth: int,
    max_iter: int,
    n_restarts: int,
    seed: int,
) -> ModelOutput:
    """Fit LG-GWR through its public pyGWRx API."""
    started = perf_counter()
    model = LGGWR(
        latent_dim=2,
        bandwidth=bandwidth if geometry == "joint" else (bandwidth, bandwidth),
        adaptive=True,
        kernel="gaussian",
        geometry=geometry,
        max_iter=max_iter,
        select_bandwidth=False,
        n_restarts=n_restarts,
        bandwidth_updates=0,
        random_state=seed,
        initialization="coordinate",
        standardize_geometry=True,
        fit_intercept=True,
    ).fit(X_train, y_train, coords_train, attrs_train)
    result = model.predict_result(X_test, coords_test, attrs_test)
    return ModelOutput(
        method=f"LG-GWR {geometry.title()}",
        predictions=result.predictions,
        coefficients=result.coefficients,
        intercepts=result.intercepts,
        fit_seconds=perf_counter() - started,
        model=model,
    )


def fit_grgwr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    coords_train: np.ndarray,
    X_test: np.ndarray,
    coords_test: np.ndarray,
    *,
    n_regimes: int,
    bandwidth: int,
    max_iter: int,
    seed: int,
    lambda_boundary: float = 1.0,
    spatial_constraint_weight: float = 0.5,
    enforce_connectivity: bool = True,
) -> ModelOutput:
    """Fit GR-GWR through its public pyGWRx API."""
    started = perf_counter()
    model = GRGWR(
        n_regimes=n_regimes,
        bandwidth=bandwidth,
        kernel="bisquare",
        lambda_boundary=lambda_boundary,
        max_iter=max_iter,
        spatial_constraint_weight=spatial_constraint_weight,
        n_neighbors=min(8, len(X_train) - 1),
        min_regime_size=max(X_train.shape[1] + 2, min(10, len(X_train) // max(2, n_regimes * 2))),
        enforce_connectivity=enforce_connectivity,
        random_state=seed,
    ).fit(X_train, y_train, coords_train)
    result = model.predict_result(X_test, coords_test)
    return ModelOutput(
        method="GR-GWR",
        predictions=result.predictions,
        coefficients=result.coefficients,
        intercepts=result.intercepts,
        fit_seconds=perf_counter() - started,
        model=model,
    )


def fit_optional_mgwr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    coords_train: np.ndarray,
    X_test: np.ndarray,
    coords_test: np.ndarray,
) -> ModelOutput:
    """Fit pyGWRx MGWR when the execution profile enables the expensive baseline."""
    started = perf_counter()
    model = MGWR(max_iter=100, search_max_iter=100, fit_intercept=True).fit(
        X_train, y_train, coords_train, compute_inference=False
    )
    predictions = model.predict(X_test, coords_test)
    return ModelOutput(
        method="MGWR",
        predictions=predictions,
        coefficients=np.full((len(X_test), X_train.shape[1]), np.nan),
        intercepts=np.full(len(X_test), np.nan),
        fit_seconds=perf_counter() - started,
        model=model,
    )


def fit_optional_sgwr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    coords_train: np.ndarray,
    X_test: np.ndarray,
    coords_test: np.ndarray,
) -> ModelOutput:
    """Fit pyGWRx SGWR using all regressors as similarity variables."""
    started = perf_counter()
    model = SGWR(
        bandwidth="aicc",
        alpha="aicc",
        similarity_vars=tuple(range(X_train.shape[1])),
        fit_intercept=True,
    ).fit(X_train, y_train, coords_train)
    result = model.predict_result(X_test, coords_test)
    return ModelOutput(
        method="SGWR",
        predictions=result.predictions,
        coefficients=result.coefficients,
        intercepts=result.intercepts,
        fit_seconds=perf_counter() - started,
        model=model,
    )
