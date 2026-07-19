"""Configuration contract tests."""

from pathlib import Path

from pygwrx_experiments.config import load_config


def test_pilot_and_full_profiles_load() -> None:
    root = Path(__file__).resolve().parents[1]
    pilot = load_config(root / "configs/pilot.json")
    full = load_config(root / "configs/full.json")
    assert pilot.profile == "pilot"
    assert full.simulation_repetitions == 100
    assert full.n_train == 400
