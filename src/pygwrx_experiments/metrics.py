"""Metrics used by simulations, spatial validation, and reporting."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, r2_score
from sklearn.neighbors import kneighbors_graph


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return MAE, RMSE, and R-squared for aligned vectors."""
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    error = pred - truth
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(r2_score(truth, pred)),
    }


def coefficient_metrics(beta_true: np.ndarray, beta_pred: np.ndarray) -> dict[str, float]:
    """Return coefficient recovery metrics over locations and predictors."""
    truth = np.asarray(beta_true, dtype=float)
    pred = np.asarray(beta_pred, dtype=float)
    if truth.shape != pred.shape:
        raise ValueError(f"Coefficient shapes differ: {truth.shape} versus {pred.shape}.")
    error = pred - truth
    correlations: list[float] = []
    for column in range(truth.shape[1]):
        if np.std(truth[:, column]) <= np.finfo(float).eps or np.std(pred[:, column]) <= np.finfo(float).eps:
            continue
        correlations.append(float(np.corrcoef(truth[:, column], pred[:, column])[0, 1]))
    return {
        "coef_mae": float(np.mean(np.abs(error))),
        "coef_rmse": float(np.sqrt(np.mean(error**2))),
        "coef_correlation": float(np.nanmean(correlations)) if correlations else math.nan,
        "sign_recovery": float(np.mean(np.sign(truth) == np.sign(pred))),
    }


def geometry_metrics(z_true: np.ndarray, z_pred: np.ndarray, *, neighbors: int = 10) -> dict[str, float]:
    """Compare true and learned pairwise geometry without requiring aligned rotations."""
    truth = np.asarray(z_true, dtype=float)
    pred = np.asarray(z_pred, dtype=float)
    if truth.shape[0] != pred.shape[0]:
        raise ValueError("True and learned coordinates must contain equal row counts.")
    true_dist = pdist(truth)
    pred_dist = pdist(pred)
    distance_correlation = float(np.corrcoef(true_dist, pred_dist)[0, 1])
    k = min(neighbors, truth.shape[0] - 1)
    true_matrix = cdist(truth, truth)
    pred_matrix = cdist(pred, pred)
    true_neighbors = np.argsort(true_matrix, axis=1)[:, 1 : k + 1]
    pred_neighbors = np.argsort(pred_matrix, axis=1)[:, 1 : k + 1]
    overlaps = []
    jaccards = []
    for left, right in zip(true_neighbors, pred_neighbors):
        left_set, right_set = set(left.tolist()), set(right.tolist())
        intersection = len(left_set & right_set)
        overlaps.append(intersection / k)
        jaccards.append(intersection / len(left_set | right_set))
    return {
        "distance_correlation": distance_correlation,
        "knn_overlap": float(np.mean(overlaps)),
        "knn_jaccard": float(np.mean(jaccards)),
    }


def undirected_knn_edges(coords: np.ndarray, neighbors: int = 8) -> set[tuple[int, int]]:
    """Build a symmetric undirected k-nearest-neighbor edge set."""
    n = len(coords)
    k = min(max(1, neighbors), n - 1)
    graph = kneighbors_graph(coords, n_neighbors=k, mode="connectivity", include_self=False)
    rows, cols = graph.nonzero()
    return {(min(int(i), int(j)), max(int(i), int(j))) for i, j in zip(rows, cols) if i != j}


def regime_metrics(labels_true: np.ndarray, labels_pred: np.ndarray, coords: np.ndarray, *, neighbors: int = 8) -> dict[str, float]:
    """Return label and graph-boundary recovery metrics."""
    truth = np.asarray(labels_true, dtype=int).reshape(-1)
    pred = np.asarray(labels_pred, dtype=int).reshape(-1)
    edges = undirected_knn_edges(coords, neighbors=neighbors)
    true_boundary = {edge for edge in edges if truth[edge[0]] != truth[edge[1]]}
    pred_boundary = {edge for edge in edges if pred[edge[0]] != pred[edge[1]]}
    intersection = len(true_boundary & pred_boundary)
    precision = intersection / len(pred_boundary) if pred_boundary else float(not true_boundary)
    recall = intersection / len(true_boundary) if true_boundary else float(not pred_boundary)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "ari": float(adjusted_rand_score(truth, pred)),
        "nmi": float(normalized_mutual_info_score(truth, pred)),
        "boundary_precision": float(precision),
        "boundary_recall": float(recall),
        "boundary_f1": float(f1),
        "true_regimes": float(np.unique(truth).size),
        "selected_regimes": float(np.unique(pred).size),
    }


def aggregate_mean(records: Iterable[dict[str, float]]) -> dict[str, float]:
    """Average numeric keys across a non-empty record sequence."""
    rows = list(records)
    if not rows:
        raise ValueError("At least one record is required.")
    keys = sorted({key for row in rows for key in row})
    return {key: float(np.nanmean([row.get(key, np.nan) for row in rows])) for key in keys}
