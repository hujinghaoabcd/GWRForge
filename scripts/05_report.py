"""Generate a Markdown report from current result tables."""

from __future__ import annotations

import argparse

from _bootstrap import ROOT
from pygwrx_experiments.reporting import generate_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="pilot")
    args = parser.parse_args()
    output = generate_report(ROOT, args.profile)
    print(output)


if __name__ == "__main__":
    main()
