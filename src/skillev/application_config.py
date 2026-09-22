"""Dependency-light public application configuration identity."""

from __future__ import annotations

from dataclasses import dataclass, field

from skillev.calibration import CalibrationConfig
from skillev.contracts import JsonValue, normalize_json, stable_hash
from skillev.diagnostics import DiagnosticsConfig
from skillev.evolution import AuthoringSamplingConfig, EvolutionConfig
from skillev.training.config import TrainerConfig

METHOD_SEMANTICS_FORMAT = "skillev-method-semantics@5"


@dataclass(frozen=True, slots=True)
class MethodSemanticsConfig:
    """The closed engineering interpretation of the method's abstract terms."""

    context_feature: str = "namespaced-task-family@1"
    calibration_outcome: str = "trajectory-terminal-success@1"
    flow_weight_normalization: str = "per-batch-invoking-edge-mean@1"
    skill_invocation_credit: str = "explicit-skill-action-in-h0@1"
    phase_detection_scope: str = "phase-search-slots-only@1"
    initial_library: str = "non-empty-seeded-library@1"
    state_flow_estimator: str = "single-observed-history-prefix@1"
    posterior_state_owner: str = "committed-projection-snapshot@1"
    posterior_skill_inheritance: str = "unchanged-ids-only-new-ids-start-at-prior@1"
    phase_no_op: str = "recheck-after-complete-fresh-comparison@2"
    feature_timing: str = "post-hoc-task-family-status-h0-tokens-final-horizon@1"
    confidence_authority: str = "evolution-k-and-explicit-split-confidence-k@1"
    source_requirements: str = "immutable-constraints-evidence-revisable-strategies@2"
    authoring_failure: str = "persist-raw-response-before-validation-stop-on-unknown@3"
    retain_infeasible: str = "abort-complete-attempt-before-authoring@1"
    generate_coverage: str = "no-explicit-invocation-public-action-including-complete@2"
    authoring_material: str = "bounded-public-execution-diversity@1"
    format: str = METHOD_SEMANTICS_FORMAT

    def __post_init__(self) -> None:
        expected = {
            "retain_infeasible": "abort-complete-attempt-before-authoring@1",
            "generate_coverage": "no-explicit-invocation-public-action-including-complete@2",
            "authoring_material": "bounded-public-execution-diversity@1",
            "source_requirements": "immutable-constraints-evidence-revisable-strategies@2",
            "authoring_failure": "persist-raw-response-before-validation-stop-on-unknown@3",
            "confidence_authority": "evolution-k-and-explicit-split-confidence-k@1",
            "feature_timing": "post-hoc-task-family-status-h0-tokens-final-horizon@1",
            "phase_no_op": "recheck-after-complete-fresh-comparison@2",
            "posterior_skill_inheritance": "unchanged-ids-only-new-ids-start-at-prior@1",
            "posterior_state_owner": "committed-projection-snapshot@1",
            "context_feature": "namespaced-task-family@1",
            "calibration_outcome": "trajectory-terminal-success@1",
            "flow_weight_normalization": "per-batch-invoking-edge-mean@1",
            "skill_invocation_credit": "explicit-skill-action-in-h0@1",
            "phase_detection_scope": "phase-search-slots-only@1",
            "initial_library": "non-empty-seeded-library@1",
            "state_flow_estimator": "single-observed-history-prefix@1",
        }
        if self.format != METHOD_SEMANTICS_FORMAT:
            raise ValueError("unsupported method semantics format")
        for field_name, expected_value in expected.items():
            if getattr(self, field_name) != expected_value:
                raise ValueError(f"unsupported method semantics: {field_name}")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "retain_infeasible": self.retain_infeasible,
            "generate_coverage": self.generate_coverage,
            "authoring_material": self.authoring_material,
            "source_requirements": self.source_requirements,
            "authoring_failure": self.authoring_failure,
            "confidence_authority": self.confidence_authority,
            "feature_timing": self.feature_timing,
            "phase_no_op": self.phase_no_op,
            "posterior_skill_inheritance": self.posterior_skill_inheritance,
            "posterior_state_owner": self.posterior_state_owner,
            "calibration_outcome": self.calibration_outcome,
            "context_feature": self.context_feature,
            "flow_weight_normalization": self.flow_weight_normalization,
            "format": self.format,
            "initial_library": self.initial_library,
            "phase_detection_scope": self.phase_detection_scope,
            "skill_invocation_credit": self.skill_invocation_credit,
            "state_flow_estimator": self.state_flow_estimator,
        }

    @classmethod
    def from_value(cls, value: object) -> MethodSemanticsConfig:
        expected_fields = {
            "retain_infeasible",
            "generate_coverage",
            "authoring_material",
            "source_requirements",
            "authoring_failure",
            "confidence_authority",
            "feature_timing",
            "phase_no_op",
            "posterior_skill_inheritance",
            "posterior_state_owner",
            "calibration_outcome",
            "context_feature",
            "flow_weight_normalization",
            "format",
            "initial_library",
            "phase_detection_scope",
            "skill_invocation_credit",
            "state_flow_estimator",
        }
        if not isinstance(value, dict) or set(value) != expected_fields:
            raise ValueError("MethodSemanticsConfig has incompatible fields")
        if any(type(item) is not str for item in value.values()):
            raise TypeError("method semantics values must be text")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class ApplicationConfig:
    trainer: TrainerConfig
    diagnostics: DiagnosticsConfig
    calibration: CalibrationConfig
    evolution: EvolutionConfig
    authoring_sampling: AuthoringSamplingConfig
    maximum_h0_tokens: int
    semantics: MethodSemanticsConfig = field(default_factory=MethodSemanticsConfig)

    def __post_init__(self) -> None:
        if type(self.maximum_h0_tokens) is not int or self.maximum_h0_tokens < 1:
            raise ValueError("maximum_h0_tokens must be positive")
        if not isinstance(self.semantics, MethodSemanticsConfig):
            raise TypeError("semantics must be MethodSemanticsConfig")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "authoring_sampling": self.authoring_sampling.to_value(),
            "calibration": self.calibration.to_value(),
            "diagnostics": self.diagnostics.to_value(),
            "evolution": self.evolution.to_value(),
            "maximum_h0_tokens": self.maximum_h0_tokens,
            "semantics": self.semantics.to_value(),
            "trainer": self.trainer.to_value(),
        }

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())

    @classmethod
    def from_value(cls, value: object) -> ApplicationConfig:
        normalized = normalize_json(value)
        fields = {
            "authoring_sampling",
            "calibration",
            "diagnostics",
            "evolution",
            "maximum_h0_tokens",
            "semantics",
            "trainer",
        }
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("ApplicationConfig has incompatible fields")
        maximum_h0_tokens = normalized["maximum_h0_tokens"]
        if type(maximum_h0_tokens) is not int:
            raise TypeError("maximum_h0_tokens must be an integer")
        return cls(
            trainer=TrainerConfig.from_value(normalized["trainer"]),
            diagnostics=DiagnosticsConfig.from_value(normalized["diagnostics"]),
            calibration=CalibrationConfig.from_value(normalized["calibration"]),
            evolution=EvolutionConfig.from_value(normalized["evolution"]),
            authoring_sampling=AuthoringSamplingConfig.from_value(normalized["authoring_sampling"]),
            maximum_h0_tokens=maximum_h0_tokens,
            semantics=MethodSemanticsConfig.from_value(normalized["semantics"]),
        )


__all__ = ["METHOD_SEMANTICS_FORMAT", "ApplicationConfig", "MethodSemanticsConfig"]
