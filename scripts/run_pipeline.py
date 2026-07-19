"""Run the complete experiment pipeline in a fail-fast order."""

from __future__ import annotations

import argparse
import subprocess
import sys

from _bootstrap import ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/pilot.json"))
    parser.add_argument("--profile", default="pilot")
    args = parser.parse_args()
    commands = [
        [sys.executable, str(ROOT / "scripts/00_environment_check.py")],
        [sys.executable, str(ROOT / "scripts/01_correctness.py")],
        [sys.executable, str(ROOT / "scripts/02_lg_simulation.py"), "--config", args.config],
        [sys.executable, str(ROOT / "scripts/03_gr_simulation.py"), "--config", args.config],
        [sys.executable, str(ROOT / "scripts/04_real_data.py"), "--config", args.config],
        [sys.executable, str(ROOT / "scripts/06_parameter_selection.py"), "--config", args.config],
        [sys.executable, str(ROOT / "scripts/07_ablation.py"), "--config", args.config],
        [sys.executable, str(ROOT / "scripts/08_benchmark.py"), "--config", args.config],
        [sys.executable, str(ROOT / "scripts/09_statistics.py")],
        [sys.executable, str(ROOT / "scripts/10_figures.py")],
        [sys.executable, str(ROOT / "scripts/05_report.py"), "--profile", args.profile],
    ]
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
