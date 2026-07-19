"""Run GR-GWR spatial-CV parameter selection."""

from __future__ import annotations

import argparse

from _bootstrap import ROOT
from pygwrx_experiments.config import load_config
from pygwrx_experiments.extended import run_gr_parameter_selection
from pygwrx_experiments.io_utils import write_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/pilot.json"))
    args = parser.parse_args()
    frame = run_gr_parameter_selection(load_config(args.config))
    write_frame(frame, ROOT / "results/gr_simulation/parameter_selection.csv")
    print(frame.sort_values("score").head(10).to_string(index=False))


if __name__ == "__main__":
    main()
