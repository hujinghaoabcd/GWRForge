"""Controlled data-generating processes for LG-GWR and GR-GWR evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LGSimulationData:
    """Training and independent prediction data for one latent-geometry scenario."""

    scenario: str
    X_train: np.ndarray
    y_train: np.ndarray
    coords_train: np.ndarray
    attrs_train: np.ndarray
    beta_train: np.ndarray
    z_train: np.ndarray | None
    X_test: np.ndarray
    y_test: np.ndarray
    coords_test: np.ndarray
    attrs_test: np.ndarray
    beta_test: np.ndarray
    z_test: np.ndarray | None


@dataclass(frozen=True)
class GRSimulationData:
    """Training and independent prediction data for one regime scenario."""

    scenario: str
    X_train: np.ndarray
    y_train: np.ndarray
    coords_train: np.ndarray
    beta_train: np.ndarray
    regimes_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    coords_test: np.ndarray
    beta_test: np.ndarray
    regimes_test: np.ndarray


def _grid(side: int) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, side)
    xx, yy = np.meshgrid(axis, axis)
    return np.column_stack([xx.ravel(), yy.ravel()])


def _scale_latent(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.vstack([train, test])
    lower = stacked.min(axis=0)
    span = stacked.max(axis=0) - lower
    span[span <= np.finfo(float).eps] = 1.0
    return (train - lower) / span, (test - lower) / span


def _latent_attributes(coords: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    u, v = coords[:, 0], coords[:, 1]
    return np.column_stack([
        np.sin(2.0 * np.pi * u) + 0.15 * rng.normal(size=len(coords)),
        np.cos(2.0 * np.pi * v) + 0.15 * rng.normal(size=len(coords)),
    ])


def make_lg_scenario(
    scenario: str,
    *,
    side: int = 20,
    n_test: int = 100,
    seed: int = 0,
    noise_sd: float = 0.5,
) -> LGSimulationData:
    """Generate one of the L1--L6 controlled latent-geometry scenarios."""
    key = scenario.upper()
    if key not in {f"L{i}" for i in range(1, 7)}:
        raise ValueError("scenario must be one of L1, ..., L6.")
    rng = np.random.default_rng(seed)
    coords_train = _grid(side)
    coords_test = rng.uniform(0.0, 1.0, size=(n_test, 2))
    attrs_train = _latent_attributes(coords_train, rng)
    attrs_test = _latent_attributes(coords_test, rng)

    if key in {"L1", "L5"}:
        z_train_raw, z_test_raw = coords_train, coords_test
    elif key == "L2":
        theta = np.deg2rad(45.0)
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        transform = rotation @ np.diag([2.5, 0.5])
        z_train_raw = coords_train @ transform.T
        z_test_raw = coords_test @ transform.T
    elif key == "L3":
        matrix = np.array([[1.2, 0.1, 0.9, -0.3], [-0.2, 0.8, 0.2, 1.1]])
        z_train_raw = np.hstack([coords_train, attrs_train]) @ matrix.T
        z_test_raw = np.hstack([coords_test, attrs_test]) @ matrix.T
    elif key == "L4":
        z_train_raw, z_test_raw = attrs_train, attrs_test
    else:
        z_train_raw, z_test_raw = coords_train, coords_test

    z_train, z_test = _scale_latent(z_train_raw, z_test_raw)
    if key == "L5":
        attrs_train = rng.normal(size=(len(coords_train), 6))
        attrs_test = rng.normal(size=(len(coords_test), 6))

    X_train = rng.normal(size=(len(coords_train), 2))
    X_test = rng.normal(size=(len(coords_test), 2))
    if key == "L6":
        z_geo_train, z_geo_test = _scale_latent(coords_train, coords_test)
        z_attr_train, z_attr_test = _scale_latent(attrs_train, attrs_test)
        beta1_train = 1.0 + 2.0 * np.sin(np.pi * z_geo_train[:, 0]) * np.cos(np.pi * z_geo_train[:, 1])
        beta1_test = 1.0 + 2.0 * np.sin(np.pi * z_geo_test[:, 0]) * np.cos(np.pi * z_geo_test[:, 1])
        beta2_train = 2.0 + z_attr_train[:, 0] ** 2 + z_attr_train[:, 1] ** 2
        beta2_test = 2.0 + z_attr_test[:, 0] ** 2 + z_attr_test[:, 1] ** 2
        geometry_train = None
        geometry_test = None
    else:
        beta1_train = 1.0 + 2.0 * np.sin(np.pi * z_train[:, 0]) * np.cos(np.pi * z_train[:, 1])
        beta1_test = 1.0 + 2.0 * np.sin(np.pi * z_test[:, 0]) * np.cos(np.pi * z_test[:, 1])
        beta2_train = 2.0 + z_train[:, 0] ** 2 + z_train[:, 1] ** 2
        beta2_test = 2.0 + z_test[:, 0] ** 2 + z_test[:, 1] ** 2
        geometry_train, geometry_test = z_train, z_test

    beta_train = np.column_stack([beta1_train, beta2_train])
    beta_test = np.column_stack([beta1_test, beta2_test])
    y_train = 3.0 + np.sum(X_train * beta_train, axis=1)
    y_test = 3.0 + np.sum(X_test * beta_test, axis=1)
    y_train += rng.normal(0.0, noise_sd, size=len(y_train))
    y_test += rng.normal(0.0, noise_sd, size=len(y_test))
    return LGSimulationData(
        scenario=key,
        X_train=X_train,
        y_train=y_train,
        coords_train=coords_train,
        attrs_train=attrs_train,
        beta_train=beta_train,
        z_train=geometry_train,
        X_test=X_test,
        y_test=y_test,
        coords_test=coords_test,
        attrs_test=attrs_test,
        beta_test=beta_test,
        z_test=geometry_test,
    )


def _regimes(scenario: str, coords: np.ndarray) -> np.ndarray:
    u, v = coords[:, 0], coords[:, 1]
    if scenario in {"R1", "R5"}:
        return np.zeros(len(coords), dtype=int)
    if scenario == "R2":
        return (u >= 0.5).astype(int)
    if scenario == "R3":
        first = 0.30 + 0.10 * np.sin(2.0 * np.pi * v)
        second = 0.68 + 0.08 * np.cos(2.0 * np.pi * v)
        return np.where(u < first, 0, np.where(u < second, 1, 2)).astype(int)
    if scenario == "R4":
        return np.where(u < 0.70, 0, np.where(u < 0.90, 1, 2)).astype(int)
    if scenario == "R6":
        return np.where((u < 0.25) | (u > 0.75), 0, 1).astype(int)
    raise ValueError("scenario must be one of R1, ..., R6.")


def _regime_betas(scenario: str, coords: np.ndarray, labels: np.ndarray) -> np.ndarray:
    u, v = coords[:, 0], coords[:, 1]
    if scenario == "R1":
        return np.column_stack([1.0 + 0.8 * u, 2.0 - 0.5 * v])
    if scenario == "R5":
        transition = np.tanh((u - 0.5) / 0.16)
        return np.column_stack([1.0 + 1.5 * transition, 1.0 - transition])
    constants = {0: (2.0, 0.5), 1: (-1.5, 1.8), 2: (0.5, -2.0)}
    beta = np.empty((len(coords), 2), dtype=float)
    for index, label in enumerate(labels):
        c1, c2 = constants[int(label)]
        beta[index] = (c1 + 0.2 * u[index] + 0.1 * v[index], c2 - 0.1 * u[index])
    return beta


def make_gr_scenario(
    scenario: str,
    *,
    side: int = 20,
    n_test: int = 100,
    seed: int = 0,
    noise_sd: float = 0.5,
) -> GRSimulationData:
    """Generate one of the R1--R6 controlled regime scenarios."""
    key = scenario.upper()
    if key not in {f"R{i}" for i in range(1, 7)}:
        raise ValueError("scenario must be one of R1, ..., R6.")
    rng = np.random.default_rng(seed)
    coords_train = _grid(side)
    coords_test = rng.uniform(0.0, 1.0, size=(n_test, 2))
    regimes_train = _regimes(key, coords_train)
    regimes_test = _regimes(key, coords_test)
    beta_train = _regime_betas(key, coords_train, regimes_train)
    beta_test = _regime_betas(key, coords_test, regimes_test)
    X_train = rng.normal(size=(len(coords_train), 2))
    X_test = rng.normal(size=(len(coords_test), 2))
    y_train = 1.0 + np.sum(X_train * beta_train, axis=1)
    y_test = 1.0 + np.sum(X_test * beta_test, axis=1)
    y_train += rng.normal(0.0, noise_sd, size=len(y_train))
    y_test += rng.normal(0.0, noise_sd, size=len(y_test))
    return GRSimulationData(
        scenario=key,
        X_train=X_train,
        y_train=y_train,
        coords_train=coords_train,
        beta_train=beta_train,
        regimes_train=regimes_train,
        X_test=X_test,
        y_test=y_test,
        coords_test=coords_test,
        beta_test=beta_test,
        regimes_test=regimes_test,
    )
