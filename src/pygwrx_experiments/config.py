"""Experiment configuration loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated execution profile shared by all experiment scripts."""

    profile: str
    random_seed: int
    grid_side: int
    n_test: int
    simulation_repetitions: int
    lg_max_iter: int
    lg_restarts: int
    gr_max_iter: int
    cv_splits: int
    cv_repeats: int
    real_datasets_lg: tuple[str, ...]
    real_datasets_gr: tuple[str, ...]
    housing_sample_size: int
    run_mgwr: bool
    run_sgwr: bool

    @property
    def n_train(self) -> int:
        """Return the number of simulation calibration observations."""
        return self.grid_side**2

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ExperimentConfig":
        """Construct and validate a configuration from decoded JSON."""
        data = dict(values)
        data["real_datasets_lg"] = tuple(data["real_datasets_lg"])
        data["real_datasets_gr"] = tuple(data["real_datasets_gr"])
        config = cls(**data)
        for name in (
            "grid_side",
            "n_test",
            "simulation_repetitions",
            "lg_restarts",
            "cv_splits",
            "cv_repeats",
            "housing_sample_size",
        ):
            if getattr(config, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if config.lg_max_iter < 0 or config.gr_max_iter < 0:
            raise ValueError("Model iteration limits must be non-negative.")
        return config


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment profile from a JSON file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        values = json.load(stream)
    return ExperimentConfig.from_dict(values)
