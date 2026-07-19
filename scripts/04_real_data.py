"""Execute repeated spatial-block validation on bundled pyGWRx datasets."""

from __future__ import annotations

import argparse

from _bootstrap import ROOT
from pygwrx_experiments.config import load_config
from pygwrx_experiments.io_utils import write_frame
from pygwrx_experiments.runners import run_real_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/pilot.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    frame = run_real_data(config, ROOT / "results/real_data")
    write_frame(frame, ROOT / "results/real_data/spatial_cv.csv")
    print(frame.groupby(["dataset", "method"])["rmse"].mean())


if __name__ == "__main__":
    main()
