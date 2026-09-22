"""Freeze an OOD evaluation architecture or select its read-only policy/library snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skillev_private.evaluation.architecture_matched import freeze_architecture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--base", required=True, type=Path)
    freeze.add_argument("--destination", required=True, type=Path)
    freeze.add_argument("--architecture-id", required=True)
    select = commands.add_parser("select")
    select.add_argument("--architecture", required=True, type=Path)
    select.add_argument("--arm-id", required=True)
    select.add_argument(
        "--snapshot", type=Path, help="Explicit policy/library settings; omit for Step-0"
    )
    select.add_argument("--reference-controls", type=Path)
    select.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze_architecture(
            json.loads(args.base.read_text()), args.destination, args.architecture_id
        )
    else:
        snapshot = json.loads(args.snapshot.read_text()) if args.snapshot else {}
        request = {
            "architecture_file": str(args.architecture.resolve()),
            "snapshot": {**snapshot, "arm_id": args.arm_id},
            "reference_controls": str(args.reference_controls.resolve())
            if args.reference_controls
            else None,
        }
        with args.output.open("x") as stream:
            json.dump(request, stream, indent=2)
            stream.write("\n")
    print(json.dumps({"generation_started": False, "operation": args.command}))


if __name__ == "__main__":
    main()
