"""Verify that the exact pyGWRx API and bundled datasets are available."""

from __future__ import annotations

import json
import platform
import sys

import numpy as np
import pandas as pd
import pygwrx
from pygwrx import GRGWR, GWR, LGGWR
from pygwrx.io import list_datasets

from _bootstrap import ROOT


def main() -> None:
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "pygwrx_version": pygwrx.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "required_classes": [GWR.__name__, LGGWR.__name__, GRGWR.__name__],
        "bundled_datasets": list_datasets(verbose=False),
        "status": "pass" if pygwrx.__version__ == "0.1.2" else "fail",
    }
    output = ROOT / "results/logs/environment.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
