#!/usr/bin/env python3
"""Validate the receipt emitted by a standard-runtime no-update control batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skillev.diagnostics.debug_controls import NoUpdateControlReceipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.receipt.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("no-update receipt must be an object")
    receipt = NoUpdateControlReceipt(**raw)
    receipt.validate()
    print("no-update control: PASS")


if __name__ == "__main__":
    main()
