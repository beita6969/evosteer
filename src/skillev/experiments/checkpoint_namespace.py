"""Private, per-attempt checkpoint namespaces.

One exact input deliberately supplies one shared *base* storage binding for
all ablation arms.  It must not cause those independent attempts to publish
the same immutable ``final-step-*`` directory.  The namespace below is a
private execution placement only: it is derived from the fixed request and
never participates in a public scientific identity.
"""

from __future__ import annotations

from pathlib import Path

from skillev.contracts import stable_hash
from skillev.runtime.attempt_protocol import AttemptBuilderKind
from skillev.training.config import PrivateCheckpointStorageBinding


def checkpoint_storage_for_attempt(
    base: PrivateCheckpointStorageBinding,
    *,
    run_id: str,
    attempt_id: str,
    builder_kind: AttemptBuilderKind,
    exact_input_sha256: str,
) -> PrivateCheckpointStorageBinding:
    """Return the one immutable checkpoint root reserved for an attempt.

    This addresses a real seven-arm collision: every arm starts from the same
    content-bound initial checkpoint, but each arm must write its own final
    optimizer/runtime snapshot.  The caller's base binding remains private;
    only a deterministic child namespace is used at runtime.
    """

    if not isinstance(base, PrivateCheckpointStorageBinding):
        raise TypeError("base must be PrivateCheckpointStorageBinding")
    if not isinstance(builder_kind, AttemptBuilderKind):
        raise TypeError("builder_kind must be AttemptBuilderKind")
    if not all(isinstance(value, str) and value for value in (run_id, attempt_id)):
        raise ValueError("run_id and attempt_id must be non-empty text")
    if not isinstance(exact_input_sha256, str) or not exact_input_sha256.startswith("sha256:"):
        raise ValueError("exact_input_sha256 must be a sha256 identity")
    namespace = stable_hash(
        {
            "attempt_id": attempt_id,
            "builder_kind": builder_kind.value,
            "exact_input_sha256": exact_input_sha256,
            "run_id": run_id,
        }
    ).removeprefix("sha256:")
    return PrivateCheckpointStorageBinding(
        directory=str(Path(base.directory).resolve() / "attempts" / namespace)
    )


__all__ = ["checkpoint_storage_for_attempt"]
