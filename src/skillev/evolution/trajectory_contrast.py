"""Same-source observational contrasts, never causal counterfactuals or votes."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from .evidence import AuthoringEdgeEvidence

from skillev.contracts import JsonValue
from skillev.contracts.observed_reset import RESET_BINDING_FORMAT, reset_binding_comparison
from skillev.rollout.artifact import RolloutArtifact

from .public_execution import PublicExecutionSnippet


@dataclass(frozen=True, slots=True)
class SourceTrajectoryIdentity:
    benchmark_id: str
    population_id: str
    source_question_id: str
    occurrence_id: str
    policy_version: str
    library_version: str
    environment_reset_identity: str = field(repr=False)
    execution_condition_id: str = ""
    reset_binding_kind: str = ""

    @property
    def condition(self) -> tuple[str, ...]:
        return (
            self.benchmark_id,
            self.population_id,
            self.source_question_id,
            self.policy_version,
            self.library_version,
            self.environment_reset_identity,
            self.execution_condition_id,
            self.reset_binding_kind,
        )

    @classmethod
    def from_artifact(cls, artifact: RolloutArtifact) -> SourceTrajectoryIdentity | None:
        payload = artifact.record.reward.native_payload
        source = payload.get("training_evidence_source") if isinstance(payload, dict) else None
        if not isinstance(source, dict):
            return None
        keys = ("benchmark_id", "population_id", "source_question_id", "occurrence_id")
        if any(not isinstance(source.get(key), str) or not source[key] for key in keys):
            return None
        # New training evidence contains the actual initial public projection
        # and observed reset. Compare its full value, not a seed-derived ID.
        kind = ""
        if "reset_binding" in source:
            comparison = reset_binding_comparison(
                source["reset_binding"], benchmark=cast(str, source["benchmark_id"])
            )
            if comparison is None:
                return None
            reset, kind = comparison
        else:
            # Preserve explicit historical evidence, but never manufacture it
            # for old records that omitted reset state.
            legacy_reset = artifact.record.initial_context.meta.get("environment_reset_identity")
            if not isinstance(legacy_reset, str) or not legacy_reset:
                return None
            reset = legacy_reset
        return cls(
            cast(str, source["benchmark_id"]),
            cast(str, source["population_id"]),
            cast(str, source["source_question_id"]),
            cast(str, source["occurrence_id"]),
            artifact.manifest.policy_snapshot.snapshot_id,
            artifact.manifest.library_version,
            reset,
            artifact.manifest.condition_id + "/" + artifact.manifest.decoding_snapshot_id,
            kind,
        )


@dataclass(frozen=True, slots=True)
class SameSourceTrajectoryContrast:
    left_source: SourceTrajectoryIdentity
    right_source: SourceTrajectoryIdentity
    left_trajectory_id: str
    right_trajectory_id: str
    shared_public_prefix: tuple[PublicExecutionSnippet, ...]
    first_divergent_step: int
    left: PublicExecutionSnippet | None
    right: PublicExecutionSnippet | None

    def __post_init__(self) -> None:
        if (
            self.left_source.condition != self.right_source.condition
            or self.left_trajectory_id == self.right_trajectory_id
        ):
            raise ValueError(
                "contrast requires distinct trajectories of one matched source condition"
            )
        if self.first_divergent_step != len(self.shared_public_prefix) + 1:
            raise ValueError("contrast divergence must follow the common public prefix")

    def authoring_value(self, *, maximum_characters_per_field: int = 800) -> dict[str, JsonValue]:
        if maximum_characters_per_field < 16:
            raise ValueError("contrast excerpt capacity too small")

        def excerpt(value: str) -> str:
            if len(value) <= maximum_characters_per_field:
                return value
            return value[: maximum_characters_per_field - 12] + " [excerpt]"

        def snippet(value: PublicExecutionSnippet | None) -> JsonValue:
            return (
                None
                if value is None
                else {
                    "action_text": excerpt(value.action_text),
                    "observation_text": excerpt(value.observation_text),
                }
            )

        source = self.left_source
        reset_fields: dict[str, JsonValue]
        if source.reset_binding_kind:
            reset_fields = {
                "reset_evidence": {
                    "format": RESET_BINDING_FORMAT,
                    "kind": source.reset_binding_kind,
                    "comparison": "exact-public-initial-state",
                    "source_field": "training_evidence_source.reset_binding",
                }
            }
        else:
            reset_fields = {"environment_reset_identity": source.environment_reset_identity}
        return {
            "format": "same-source-observational-contrast@2"
            if source.reset_binding_kind
            else "same-source-observational-contrast@1",
            "benchmark_id": source.benchmark_id,
            "population_id": source.population_id,
            "source_question_id": source.source_question_id,
            "left_occurrence_id": self.left_source.occurrence_id,
            "right_occurrence_id": self.right_source.occurrence_id,
            "policy_version": source.policy_version,
            "library_version": source.library_version,
            **reset_fields,
            "execution_condition_id": source.execution_condition_id,
            "left_trajectory_id": self.left_trajectory_id,
            "right_trajectory_id": self.right_trajectory_id,
            "first_divergent_step": self.first_divergent_step,
            "shared_prefix_length": len(self.shared_public_prefix),
            "left": snippet(self.left),
            "right": snippet(self.right),
            "left_complete_result_reference": "trajectory:" + self.left_trajectory_id,
            "right_complete_result_reference": "trajectory:" + self.right_trajectory_id,
            "selection": "same-full-source-condition-then-lexicographic-trajectory-pair",
            "excerpt_rule": "public-action-observation-prefix-per-field@1",
            "omitted": [
                "reasoning",
                "private-reward-payload",
                "hidden-tests",
                "rubrics",
                "shared-prefix-text",
                "post-divergence-tail",
            ],
            "causal_claim": False,
        }


def same_source_contrasts(
    artifacts: tuple[RolloutArtifact, ...], *, maximum_pairs: int = 16
) -> tuple[SameSourceTrajectoryContrast, ...]:
    if type(maximum_pairs) is not int or maximum_pairs < 1:
        raise ValueError("contrast capacity must be a positive integer")
    groups: dict[tuple[str, ...], list[tuple[SourceTrajectoryIdentity, RolloutArtifact]]] = {}
    seen: set[str] = set()
    for artifact in artifacts:
        if artifact.record.trajectory_id in seen:
            raise ValueError("contrast input repeats a trajectory")
        seen.add(artifact.record.trajectory_id)
        identity = SourceTrajectoryIdentity.from_artifact(artifact)
        if identity is not None:
            groups.setdefault(identity.condition, []).append((identity, artifact))
    result = []
    for condition in sorted(groups):
        group = sorted(groups[condition], key=lambda item: item[1].record.trajectory_id)
        for (left_source, left), (right_source, right) in combinations(group, 2):
            lsteps = tuple(PublicExecutionSnippet.from_step(step) for step in left.record.steps)
            rsteps = tuple(PublicExecutionSnippet.from_step(step) for step in right.record.steps)
            shared = 0
            while shared < min(len(lsteps), len(rsteps)) and lsteps[shared] == rsteps[shared]:
                shared += 1
            if shared == len(lsteps) == len(rsteps):
                continue
            result.append(
                SameSourceTrajectoryContrast(
                    left_source,
                    right_source,
                    left.record.trajectory_id,
                    right.record.trajectory_id,
                    lsteps[:shared],
                    shared + 1,
                    lsteps[shared] if shared < len(lsteps) else None,
                    rsteps[shared] if shared < len(rsteps) else None,
                )
            )
            if len(result) == maximum_pairs:
                return tuple(result)
    return tuple(result)


def attach_source_contrasts(
    artifacts: tuple[RolloutArtifact, ...], evidence: tuple[AuthoringEdgeEvidence, ...]
) -> tuple[AuthoringEdgeEvidence, ...]:
    """Enrich only exact matching committed public edge evidence; no Φ decision."""
    from dataclasses import replace

    from skillev.contracts import canonical_json

    by_edge: dict[str, list[str]] = {}
    for contrast in same_source_contrasts(artifacts):
        encoded = canonical_json(contrast.authoring_value())
        for trajectory in (contrast.left_trajectory_id, contrast.right_trajectory_id):
            by_edge.setdefault(f"{trajectory}:{contrast.first_divergent_step}", []).append(encoded)
    return tuple(
        replace(
            item,
            public_execution=replace(
                item.public_execution, contrasts_json=tuple(by_edge[item.edge_id])
            ),
        )
        if item.edge_id in by_edge and item.public_execution is not None
        else item
        for item in evidence
    )
