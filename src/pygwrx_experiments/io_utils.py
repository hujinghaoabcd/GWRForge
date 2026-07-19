"""Small file-system helpers for deterministic experiment outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory and return a normalized path."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def write_frame(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a CSV with stable column ordering and no index."""
    output = ensure_parent(path)
    frame.to_csv(output, index=False)
    return output


def write_json(data: dict[str, Any], path: str | Path) -> Path:
    """Write UTF-8 JSON with deterministic formatting."""
    output = ensure_parent(path)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
    return output
