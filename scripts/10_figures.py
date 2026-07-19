"""Generate summary figures from completed result tables."""

from __future__ import annotations

from _bootstrap import ROOT
from pygwrx_experiments.figures import generate_all


def main() -> None:
    for output in generate_all(ROOT):
        print(output)


if __name__ == "__main__":
    main()
