#!/usr/bin/env python3
"""Compatibility notice for the split SWE generation/evaluation workflow."""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "SWE generation and scoring are deliberately separate. Run "
        "generate_qwen35_direct_swe.py, then score_qwen35_direct_swe.py."
    )


if __name__ == "__main__":
    main()
