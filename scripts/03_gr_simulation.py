"""Execute GR-GWR R1--R6 simulations."""

from __future__ import annotations

import argparse

from _bootstrap import ROOT
from pygwrx_experiments.config import load_config
from pygwrx_experiments.io_utils import write_frame
from pygwrx_experiments.runners import run_gr_simulations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/pilot.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    frame = run_gr_simulations(config)
    write_frame(frame, ROOT / "results/gr_simulation/gr_simulation.csv")
    print(frame.groupby(["scenario", "method"])["rmse"].mean())


if __name__ == "__main__":
    main()
