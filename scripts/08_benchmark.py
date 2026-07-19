"""Run size-versus-runtime benchmarks."""

from __future__ import annotations

import argparse

from _bootstrap import ROOT
from pygwrx_experiments.config import load_config
from pygwrx_experiments.extended import run_runtime_benchmark
from pygwrx_experiments.io_utils import write_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/pilot.json"))
    args = parser.parse_args()
    frame = run_runtime_benchmark(load_config(args.config))
    write_frame(frame, ROOT / "results/logs/runtime_benchmark.csv")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
