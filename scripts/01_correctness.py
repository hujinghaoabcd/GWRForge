"""Execute required numerical degeneration checks."""

from __future__ import annotations

from _bootstrap import ROOT
from pygwrx_experiments.correctness import run_correctness_checks
from pygwrx_experiments.io_utils import write_frame


def main() -> None:
    frame = run_correctness_checks()
    write_frame(frame, ROOT / "results/correctness/correctness.csv")
    print(frame.to_string(index=False))
    if not frame["pass"].all():
        raise SystemExit("At least one numerical degeneration check failed.")


if __name__ == "__main__":
    main()
