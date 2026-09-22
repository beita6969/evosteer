"""Canonical, result-blind protocol artifact construction.

The private schedule producer supplies only public schedule identities here;
task IDs stay in its private frozen JSON.  This keeps the tracked protocol
wire answer-free while preventing hand-edited protocol/freeze hashes.
"""

from __future__ import annotations

import os
from pathlib import Path

from skillev.contracts import JsonValue, canonical_json

from .formal_execution import FormalExecutionFreeze
from .protocol import (
    ABLATION_PROTOCOLS,
    BENCHMARK_SPECS,
    DatasetSnapshotIdentity,
    ExperimentProtocol,
    FrozenTaskSequenceIdentity,
    ProtocolFreeze,
    create_protocol,
)


def build_preregistered_protocol(
    *,
    dataset_snapshots: tuple[DatasetSnapshotIdentity, ...],
    schedule_identities: tuple[FrozenTaskSequenceIdentity, ...],
    formal_execution: FormalExecutionFreeze,
) -> ExperimentProtocol:
    """Build the only result-blind protocol from public frozen identities."""

    return create_protocol(
        dataset_snapshots=dataset_snapshots,
        schedule_identities=schedule_identities,
        formal_execution=formal_execution,
    )


def protocol_artifact_values(
    protocol: ExperimentProtocol,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    """Return canonical protocol/freeze values after all public invariants hold."""

    if not isinstance(protocol, ExperimentProtocol):
        raise TypeError("protocol artifact construction requires ExperimentProtocol")
    if protocol.benchmarks != BENCHMARK_SPECS or protocol.ablations != ABLATION_PROTOCOLS:
        raise ValueError("protocol artifact differs from the preregistered benchmark or arm order")
    freeze = ProtocolFreeze.create(protocol)
    freeze_value = freeze.to_value()
    if freeze_value["result_ingestion_count"] != 0:
        raise ValueError("protocol freeze must precede all result ingestion")
    return protocol.to_value(), freeze_value


def write_preregistered_protocol_artifacts(
    *,
    protocol: ExperimentProtocol,
    protocol_path: Path,
    freeze_path: Path,
) -> ProtocolFreeze:
    """Write new canonical artifacts once; callers never edit hashes manually."""

    if not isinstance(protocol_path, Path) or not isinstance(freeze_path, Path):
        raise TypeError("protocol artifact paths must be Path values")
    if protocol_path.exists() or freeze_path.exists():
        raise FileExistsError("protocol artifacts are write-once; publish to fresh paths")
    protocol_value, freeze_value = protocol_artifact_values(protocol)
    _write_once(protocol_path, protocol_value)
    _write_once(freeze_path, freeze_value)
    return ProtocolFreeze.from_value(freeze_value)


def _write_once(path: Path, value: dict[str, JsonValue]) -> None:
    parent = path.parent
    if not parent.is_dir():
        raise NotADirectoryError(parent)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_json(value))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


__all__ = [
    "build_preregistered_protocol",
    "protocol_artifact_values",
    "write_preregistered_protocol_artifacts",
]
