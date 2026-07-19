"""Run LG-GWR and GR-GWR ablation studies."""

from __future__ import annotations

import argparse

from _bootstrap import ROOT
from pygwrx_experiments.config import load_config
from pygwrx_experiments.extended import run_gr_ablation, run_lg_ablation
from pygwrx_experiments.io_utils import write_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/pilot.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    lg = run_lg_ablation(config)
    gr = run_gr_ablation(config)
    write_frame(lg, ROOT / "results/lg_simulation/ablation.csv")
    write_frame(gr, ROOT / "results/gr_simulation/ablation.csv")
    print("LG ablation\n", lg[["variant", "rmse", "coef_rmse"]].to_string(index=False))
    print("GR ablation\n", gr[["variant", "rmse", "ari", "boundary_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
