"""Private materialization identity for public Protocol 11."""

from __future__ import annotations

from pathlib import Path

from skillev.experiments.protocol_v11 import ProtocolV11Spec, load_protocol_v11


def load_repository_protocol_v11(repository_root: Path) -> ProtocolV11Spec:
    if not repository_root.is_absolute() or not repository_root.is_dir():
        raise ValueError("repository root must be an absolute directory")
    return load_protocol_v11(
        repository_root / "configs/evaluation/protocol_v11.yaml",
        repository_root / "configs/evaluation/protocol_v11_sources.yaml",
    )


__all__ = ["load_repository_protocol_v11"]
