"""Run paired Wilcoxon comparisons against GWR with Holm correction."""

from __future__ import annotations

import pandas as pd

from _bootstrap import ROOT
from pygwrx_experiments.extended import paired_statistics
from pygwrx_experiments.io_utils import write_frame


def main() -> None:
    jobs = [
        (ROOT / "results/lg_simulation/lg_simulation.csv", "scenario", ROOT / "results/lg_simulation/paired_statistics.csv"),
        (ROOT / "results/gr_simulation/gr_simulation.csv", "scenario", ROOT / "results/gr_simulation/paired_statistics.csv"),
        (ROOT / "results/real_data/spatial_cv.csv", "dataset", ROOT / "results/real_data/paired_statistics.csv"),
    ]
    for source, group, output in jobs:
        if source.exists():
            frame = paired_statistics(pd.read_csv(source), group)
            write_frame(frame, output)
            print(output, len(frame))


if __name__ == "__main__":
    main()
