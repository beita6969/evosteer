"""Persist quality decisions and request the existing cooperative checkpoint stop.

Probe generation is independent, read-only work. The training owner calls this
only at its normal transaction boundary. A missing due probe pauses rather than
advancing the next batch under unverified quality. Existing checkpoints and
native labels are never changed here.
"""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from pathlib import Path

from .quality_gate import ProtocolProbe, QualityGatePolicy, QualityRule, evaluate_quality


def _write(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_quality_policy(path: Path) -> QualityGatePolicy:
    value = json.loads(path.read_text(encoding="utf-8"))
    value["rules"] = tuple(QualityRule(**rule) for rule in value["rules"])
    return QualityGatePolicy(**value)


class QualityCheckpointStop:
    def __init__(self, root: Path, policy: QualityGatePolicy) -> None:
        self.root = root
        self.policy = policy
        self.root.mkdir(parents=True, exist_ok=True)
        saved = root / "policy.json"
        value = json.loads(json.dumps(asdict(policy)))
        if saved.exists():
            # Optional policy fields may be absent from an older persisted rule.
            # Compare their interpreted defaults without rewriting that history.
            if json.loads(json.dumps(asdict(load_quality_policy(saved)))) != value:
                raise ValueError("quality rules changed during the run; declare a new condition")
        else:
            _write(saved, value)

    def _probe(self, step: int) -> ProtocolProbe | None:
        path = self.root / f"probe-{step:08d}.json"
        if not path.exists():
            return None
        probe = ProtocolProbe(**json.loads(path.read_text(encoding="utf-8")))
        if probe.policy_step != step:
            raise ValueError("probe file targets a different policy step")
        return probe

    def check(
        self, *, policy_step: int, policy_snapshot_id: str, library_snapshot_id: str | None = None
    ) -> bool:
        probes = tuple(
            p
            for step in range(self.policy.cadence, policy_step + 1, self.policy.cadence)
            if step > self.policy.baseline_step and (p := self._probe(step)) is not None
        )
        decision = evaluate_quality(
            self.policy,
            baseline=self._probe(self.policy.baseline_step),
            probes=probes,
            policy_step=policy_step,
            policy_snapshot_id=policy_snapshot_id,
            library_snapshot_id=library_snapshot_id,
        )
        value = decision.to_value()
        # Append only distinct decisions: resumption with a newly available
        # probe retains the earlier missing/paused evidence instead of erasing it.
        path = self.root / f"decisions-{policy_step:08d}.json"
        history = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        if not history or history[-1] != value:
            history.append(value)
            _write(path, history)
        return decision.action == "pause-after-commit"


class QualityProbeCoordinator:
    """Await one isolated probe at a due, fully committed training boundary.

    The collector persists its result in its own directory before returning.
    Recovery installs that completed result without generating another answer.
    An interrupted attempt without a complete result remains unverified; it is
    not permission to resample. No production budget or evidence is accepted.
    """

    def __init__(
        self,
        monitor: QualityCheckpointStop,
        collect: Callable[[Path, int, str], Awaitable[ProtocolProbe]],
    ) -> None:
        self.monitor, self.collect = monitor, collect

    async def check(
        self, *, policy_step: int, policy_snapshot_id: str, library_snapshot_id: str | None = None
    ) -> bool:
        monitor = self.monitor
        if (
            policy_step != monitor.policy.baseline_step and policy_step % monitor.policy.cadence
        ) or monitor._probe(policy_step) is not None:
            return monitor.check(
                policy_step=policy_step,
                policy_snapshot_id=policy_snapshot_id,
                library_snapshot_id=library_snapshot_id,
            )
        name = f"probe-{policy_step:08d}"
        directory = monitor.root / "collections" / name
        completed = directory / f"{name}.json"
        attempt = monitor.root / f"attempt-{policy_step:08d}.json"
        probe = None
        if completed.is_file():
            probe = ProtocolProbe(**json.loads(completed.read_text(encoding="utf-8")))
        elif not attempt.exists():
            _write(
                attempt,
                {
                    "status": "collecting",
                    "policy_step": policy_step,
                    "policy_snapshot_id": policy_snapshot_id,
                },
            )
            try:
                probe = await self.collect(directory, policy_step, policy_snapshot_id)
            except (OSError, TimeoutError, RuntimeError) as error:
                _write(
                    attempt,
                    {
                        "status": "probe-infrastructure-failure",
                        "error_type": type(error).__name__,
                        "policy_step": policy_step,
                        "policy_snapshot_id": policy_snapshot_id,
                    },
                )
                # The already committed training update is preserved. A missing
                # native verdict is never turned into success=False or a pass.
                # Unexpected programming/identity errors propagate instead of
                # being disguised as an ordinary unavailable measurement.
        if probe is not None:
            if (
                probe.policy_step != policy_step
                or probe.policy_snapshot_id != policy_snapshot_id
                or probe.panel_id != monitor.policy.panel_id
                or probe.condition_id != monitor.policy.condition_id
            ):
                raise ValueError("completed quality probe targets another frozen condition")
            _write(monitor.root / f"{name}.json", asdict(probe))
            _write(attempt, {"status": "complete", "evidence_id": probe.evidence_id})
        return monitor.check(
            policy_step=policy_step,
            policy_snapshot_id=policy_snapshot_id,
            library_snapshot_id=library_snapshot_id,
        )
