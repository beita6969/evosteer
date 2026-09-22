"""Answer-free evidence coordinates and actual H0 exposure, not skill credit.

Source-question identities come only from the trusted training binding. Missing
historical identities stay unknown; neither task-ID suffixes nor query text are
used to manufacture independent questions. These fields never reach an actor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from skillev.contracts import JsonValue
from skillev.contracts.skill_exposure import CATALOG_EXPOSURES, SKILL_EXPOSURES
from skillev.rollout.context import INITIAL_CONTEXT_FORMAT_VERSION

from .invocation_evidence import InvocationExecutionLink, invocation_execution_links

if TYPE_CHECKING:
    from skillev.rollout import RolloutArtifact


@dataclass(frozen=True, slots=True)
class TrajectoryEvidenceContext:
    trajectory_id: str
    task_family: str
    benchmark_id: str | None
    source_population_id: str | None
    source_question_id: str | None
    matched_skill_ids: tuple[str, ...] | None
    retrieved_skill_ids: tuple[str, ...]
    visible_skill_ids: tuple[str, ...] | None
    active_skill_ids: tuple[str, ...]
    executed_edge_count: int
    invoking_edge_count: int
    invocation_links: tuple[InvocationExecutionLink, ...] | None = None
    skill_exposure: str | None = None
    skill_input_evidence: tuple[dict[str, JsonValue], ...] | None = None
    terminal_verifier_version: str | None = None
    context_id: str | None = None
    available_tools: tuple[str, ...] | None = None
    terminal_success: bool | None = None
    terminal_reward: float | None = None

    def to_value(self) -> dict[str, JsonValue]:
        return {
            **(
                cast(
                    dict[str, JsonValue],
                    {
                        "context_id": self.context_id,
                        "available_tools": list(self.available_tools or ()),
                    },
                )
                if self.context_id is not None
                else {}
            ),
            **(
                {"terminal_success": self.terminal_success, "terminal_reward": self.terminal_reward}
                if self.terminal_success is not None
                else {}
            ),
            **(
                {"skill_input_evidence": cast(JsonValue, list(self.skill_input_evidence))}
                if self.skill_input_evidence is not None
                else {}
            ),
            **(
                {"terminal_verifier_version": self.terminal_verifier_version}
                if self.terminal_verifier_version is not None
                else {}
            ),
            **({"skill_exposure": self.skill_exposure} if self.skill_exposure is not None else {}),
            "invocation_links": [item.to_value() for item in self.invocation_links]
            if self.invocation_links is not None
            else None,
            "trajectory_id": self.trajectory_id,
            "task_family": self.task_family,
            "benchmark_id": self.benchmark_id,
            "source_population_id": self.source_population_id,
            "source_question_id": self.source_question_id,
            "matched_skill_ids": None
            if self.matched_skill_ids is None
            else list(self.matched_skill_ids),
            "retrieved_skill_ids": list(self.retrieved_skill_ids),
            "visible_skill_ids": None
            if self.visible_skill_ids is None
            else list(self.visible_skill_ids),
            "active_skill_ids": list(self.active_skill_ids),
            "executed_edge_count": self.executed_edge_count,
            "invoking_edge_count": self.invoking_edge_count,
        }

    @classmethod
    def from_value(cls, value: object) -> TrajectoryEvidenceContext:
        if not isinstance(value, dict):
            raise TypeError("trajectory evidence context must be an object")
        return cls(
            context_id=value.get("context_id"),
            available_tools=tuple(value["available_tools"])
            if value.get("available_tools") is not None
            else None,
            terminal_success=value.get("terminal_success"),
            terminal_reward=value.get("terminal_reward"),
            skill_input_evidence=tuple(value["skill_input_evidence"])
            if value.get("skill_input_evidence") is not None
            else None,
            terminal_verifier_version=value.get("terminal_verifier_version"),
            skill_exposure=value.get("skill_exposure"),
            invocation_links=tuple(
                InvocationExecutionLink.from_value(item) for item in value["invocation_links"]
            )
            if value.get("invocation_links") is not None
            else None,
            trajectory_id=value["trajectory_id"],
            task_family=value["task_family"],
            benchmark_id=value["benchmark_id"],
            source_population_id=value["source_population_id"],
            source_question_id=value["source_question_id"],
            matched_skill_ids=None
            if value["matched_skill_ids"] is None
            else tuple(value["matched_skill_ids"]),
            retrieved_skill_ids=tuple(value["retrieved_skill_ids"]),
            visible_skill_ids=None
            if value["visible_skill_ids"] is None
            else tuple(value["visible_skill_ids"]),
            active_skill_ids=tuple(value["active_skill_ids"]),
            executed_edge_count=value["executed_edge_count"],
            invoking_edge_count=value["invoking_edge_count"],
        )

    @property
    def body_visible_skill_ids(self) -> tuple[str, ...] | None:
        if self.skill_exposure == "full-inline":
            return self.visible_skill_ids
        if self.skill_exposure in CATALOG_EXPOSURES:
            if self.skill_input_evidence is None:
                return None
            rows = [row for row in self.skill_input_evidence if row.get("phase") == "action"]
            if any(not isinstance(row.get("visible_skill_body_refs"), list) for row in rows):
                return None
            return tuple(
                sorted(
                    {
                        str(ref["skill_id"])
                        for row in rows
                        for ref in cast(list[JsonValue], row["visible_skill_body_refs"])
                        if isinstance(ref, dict)
                    }
                )
            )
        return None

    @property
    def canonical_source_key(self) -> tuple[str, str] | None:
        if self.benchmark_id and self.source_question_id:
            return self.benchmark_id, self.source_question_id
        return None

    @property
    def source_key(self) -> tuple[str, str, str] | None:
        if self.benchmark_id and self.source_population_id and self.source_question_id:
            return self.benchmark_id, self.source_population_id, self.source_question_id
        return None

    @classmethod
    def from_artifact(cls, artifact: RolloutArtifact) -> TrajectoryEvidenceContext:
        record = artifact.record
        h0 = artifact.initial_context.contract
        raw = record.reward.native_payload.get("training_evidence_source")
        source = raw if isinstance(raw, dict) else {}

        def text(key: str) -> str | None:
            value = source.get(key)
            return value if isinstance(value, str) and value else None

        inclusions = h0.meta.get("retrieval_inclusions")
        matched = (
            tuple(
                str(item["skill_id"])
                for item in inclusions
                if isinstance(item, dict) and item.get("inclusion_reason") == "applicability-match"
            )
            if isinstance(inclusions, list | tuple)
            else None
        )
        # A visible catalog ID is not exposure to its body and is not an invocation.
        # The exposure condition and execution links preserve that distinction.
        return cls(
            context_id=str(h0.meta["context_id"])
            if h0.meta.get("context_id") is not None
            else None,
            available_tools=tuple(str(t) for t in cast(list[JsonValue], h0.meta["available_tools"]))
            if isinstance(h0.meta.get("available_tools"), tuple | list)
            else None,
            terminal_success=record.reward.success,
            terminal_reward=record.reward.value,
            skill_exposure=str(h0.meta.get("skill_exposure", "full-inline"))
            if h0.assembler_version == INITIAL_CONTEXT_FORMAT_VERSION
            or h0.meta.get("skill_exposure") in SKILL_EXPOSURES
            else None,
            invocation_links=invocation_execution_links(record, artifact.skill_input_evidence),
            skill_input_evidence=artifact.skill_input_evidence,
            terminal_verifier_version=record.reward.verifier_version,
            trajectory_id=record.trajectory_id,
            task_family=str(h0.meta.get("task_family", "unknown")),
            benchmark_id=text("benchmark_id"),
            source_population_id=text("population_id"),
            source_question_id=text("source_question_id"),
            matched_skill_ids=matched,
            retrieved_skill_ids=h0.retrieved_skill_ids,
            visible_skill_ids=(
                tuple(
                    sorted(
                        {
                            str(skill)
                            for row in artifact.skill_input_evidence
                            for skill in cast(list[JsonValue], row["catalog_visible_skill_ids"])
                        }
                    )
                )
                if artifact.skill_input_evidence is not None
                and all(
                    isinstance(row.get("catalog_visible_skill_ids"), list)
                    for row in artifact.skill_input_evidence
                )
                else None
            )
            if h0.meta.get("skill_exposure") in CATALOG_EXPOSURES
            else (
                h0.retrieved_skill_ids
                if h0.assembler_version == INITIAL_CONTEXT_FORMAT_VERSION
                or h0.meta.get("skill_exposure") in SKILL_EXPOSURES
                else None
            ),
            active_skill_ids=h0.active_skill_ids,
            executed_edge_count=record.horizon,
            invoking_edge_count=sum(bool(step.invoked_skill_ids) for step in record.steps),
        )
