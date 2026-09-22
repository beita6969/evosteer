#!/usr/bin/env python3
"""Render a public Markdown table from answer-free machine receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skillev.evaluation.current_iid.rendering import (
    render_current_iid_markdown,
    render_protocol_v12_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = json.loads(args.aggregate.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("aggregate must be a JSON object")
    renderer = (
        render_protocol_v12_markdown
        if value.get("format") == "skillev-current-iid-result@2"
        else render_current_iid_markdown
    )
    args.output.write_text(renderer(value), encoding="utf-8")


if __name__ == "__main__":
    main()
