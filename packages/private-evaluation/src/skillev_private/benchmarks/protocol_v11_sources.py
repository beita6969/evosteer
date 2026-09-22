"""Private Protocol 11 source projections with explicit public/private boundaries."""

from __future__ import annotations

from dataclasses import dataclass

from skillev.contracts import JsonValue, normalize_json
from skillev.experiments.protocol_v11 import BenchmarkV11, PopulationRoleV11
from skillev.rollout import RolloutTask


@dataclass(frozen=True, slots=True)
class ProtocolV11PrivateRecord:
    benchmark: BenchmarkV11
    role: PopulationRoleV11
    source_id: str
    task: RolloutTask
    private_payload: dict[str, JsonValue]

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("Protocol 11 source ID must be non-empty")
        normalized = normalize_json(self.private_payload)
        if not isinstance(normalized, dict):
            raise ValueError("Protocol 11 private payload must be an object")
        object.__setattr__(self, "private_payload", normalized)
        forbidden_public = {
            "accepted_answers",
            "rubrics",
            "reference_response",
            "tests",
            "gold_patch",
            "golden_workbook",
            "oracle",
        }
        public = self.task.public_context
        if isinstance(public, dict) and forbidden_public.intersection(public):
            raise ValueError("Protocol 11 task exposes evaluator-private fields")


def validate_population_disjointness(
    records: tuple[ProtocolV11PrivateRecord, ...],
) -> None:
    by_benchmark: dict[BenchmarkV11, dict[PopulationRoleV11, set[str]]] = {}
    for record in records:
        roles = by_benchmark.setdefault(record.benchmark, {})
        ids = roles.setdefault(record.role, set())
        if record.source_id in ids:
            raise ValueError("duplicate source ID inside Protocol 11 population")
        ids.add(record.source_id)
    for roles in by_benchmark.values():
        training = roles.get(PopulationRoleV11.TRAINING, set())
        validation = roles.get(PopulationRoleV11.VALIDATION, set())
        final = roles.get(PopulationRoleV11.FINAL_EVALUATION, set())
        if training & validation or training & final or validation & final:
            raise ValueError("Protocol 11 training/validation/final source IDs overlap")


__all__ = ["ProtocolV11PrivateRecord", "validate_population_disjointness"]
