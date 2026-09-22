#!/usr/bin/env python3
"""Build the immutable source archive consumed by Gate 4c @6."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.build_gate_4a_source_package import build_source_package


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = _args()
    build_source_package(
        repository=Path(__file__).resolve().parents[1],
        output=Path(args.output),
    )


if __name__ == "__main__":
    main()
