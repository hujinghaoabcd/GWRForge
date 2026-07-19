"""Numerical degeneration checks required before substantive experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pygwrx import GRGWR, GWR, LGGWR


def run_correctness_checks(seed: int = 20260719) -> pd.DataFrame:
    """Run LG-GWR-to-GWR and one-regime GR-GWR-to-GWR checks."""
    rng = np.random.default_rng(seed)
    n = 49
    coords = rng.uniform(0.0, 10.0, size=(n, 2))
    X = rng.normal(size=(n, 2))
    y = 1.0 + (1.2 + 0.08 * coords[:, 0]) * X[:, 0] - 0.6 * X[:, 1]
    y += rng.normal(0.0, 0.08, size=n)

    rows: list[dict[str, object]] = []

    fixed_bandwidth = 4.0
    gwr_fixed = GWR(kernel="gaussian", bandwidth=fixed_bandwidth, adaptive=False, fit_intercept=True).fit(X, y, coords)
    lggwr = LGGWR(
        geometry="separable",
        bandwidth=(fixed_bandwidth, 1.0e12),
        adaptive=False,
        kernel="gaussian",
        max_iter=0,
        select_bandwidth=False,
        bandwidth_updates=0,
        standardize_geometry=False,
        random_state=seed,
        fit_intercept=True,
    ).fit(X, y, coords, np.zeros((n, 1)))
    coef_diff = float(np.max(np.abs(lggwr.coefficients_ - np.column_stack([gwr_fixed.intercept_, gwr_fixed.coef_]))))
    fitted_diff = float(np.max(np.abs(lggwr.fitted_values_ - gwr_fixed.fitted_values_)))
    aicc_diff = float(abs(lggwr.diagnostics_["aicc"] - gwr_fixed.diagnostics_["aicc"]))
    rows.append({
        "check": "LG-GWR separable -> fixed geographic GWR",
        "max_coef_diff": coef_diff,
        "max_fitted_diff": fitted_diff,
        "aicc_diff": aicc_diff,
        "tolerance": 1.0e-6,
        "pass": coef_diff < 1.0e-6 and fitted_diff < 1.0e-6,
    })

    adaptive_bandwidth = 18
    gwr_adaptive = GWR(kernel="bisquare", bandwidth=adaptive_bandwidth, adaptive=True, fit_intercept=True).fit(X, y, coords)
    grgwr = GRGWR(
        n_regimes=1,
        bandwidth=adaptive_bandwidth,
        kernel="bisquare",
        max_iter=0,
        n_neighbors=6,
        random_state=seed,
        fit_intercept=True,
    ).fit(X, y, coords)
    gr_params = grgwr.local_parameters_
    gwr_params = np.column_stack([gwr_adaptive.intercept_, gwr_adaptive.coef_])
    coef_diff = float(np.max(np.abs(gr_params - gwr_params)))
    fitted_diff = float(np.max(np.abs(grgwr.fitted_values_ - gwr_adaptive.fitted_values_)))
    aicc_diff = float(abs(grgwr.diagnostics_["aicc"] - gwr_adaptive.diagnostics_["aicc"]))
    rows.append({
        "check": "GR-GWR one regime -> adaptive GWR",
        "max_coef_diff": coef_diff,
        "max_fitted_diff": fitted_diff,
        "aicc_diff": aicc_diff,
        "tolerance": 1.0e-6,
        "pass": coef_diff < 1.0e-6 and fitted_diff < 1.0e-6,
    })
    return pd.DataFrame(rows)
