"""Fast structural tests for controlled data generators and metrics."""

import numpy as np

from pygwrx_experiments.metrics import geometry_metrics, regime_metrics
from pygwrx_experiments.simulations import make_gr_scenario, make_lg_scenario


def test_lg_scenarios_are_aligned() -> None:
    for scenario in (f"L{i}" for i in range(1, 7)):
        data = make_lg_scenario(scenario, side=5, n_test=12, seed=3)
        assert data.X_train.shape == (25, 2)
        assert data.beta_train.shape == (25, 2)
        assert data.attrs_train.shape[0] == 25
        assert data.y_test.shape == (12,)


def test_gr_scenarios_are_aligned() -> None:
    for scenario in (f"R{i}" for i in range(1, 7)):
        data = make_gr_scenario(scenario, side=5, n_test=12, seed=4)
        assert data.X_train.shape == (25, 2)
        assert data.regimes_train.shape == (25,)
        assert data.beta_test.shape == (12, 2)


def test_recovery_metrics_are_one_for_identical_structures() -> None:
    coords = np.column_stack([np.arange(12), np.zeros(12)])
    labels = np.repeat([0, 1], 6)
    regime = regime_metrics(labels, labels, coords, neighbors=2)
    geometry = geometry_metrics(coords, coords, neighbors=3)
    assert regime["ari"] == 1.0
    assert regime["boundary_f1"] == 1.0
    assert np.isclose(geometry["distance_correlation"], 1.0)
    assert np.isclose(geometry["knn_jaccard"], 1.0)
