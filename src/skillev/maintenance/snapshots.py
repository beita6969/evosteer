"""Explicit snapshot maintenance, never imported by method core."""

from __future__ import annotations

import shutil
from pathlib import Path


def list_orphan_staging(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for path in root.glob(".*.staging-*") if path.is_dir()))


def delete_exact_paths(paths: tuple[Path, ...]) -> None:
    for path in paths:
        resolved = path.resolve()
        if not resolved.name.startswith(".") or ".staging-" not in resolved.name:
            raise ValueError(f"not an orphan staging directory: {resolved}")
        shutil.rmtree(resolved)


def prune_exact_step_snapshots(*, root: Path, delete_names: tuple[str, ...]) -> None:
    for name in delete_names:
        if not name.startswith("step-") or Path(name).name != name:
            raise ValueError(f"invalid explicit step snapshot name: {name!r}")
        shutil.rmtree(root / name)
