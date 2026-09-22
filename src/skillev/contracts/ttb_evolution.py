"""Closed Protocol-v3 contracts for the five skill-evolution actions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, TypeAlias

from .canonical import JsonValue, normalize_json, stable_hash
from .identity import validate_sha256
from .ttb_calibration import ContextFeature
from .ttb_common import require_finite_number, require_non_empty_text

EVOLUTION_FORMAT = "skillev-evolution@5"


class EvolutionActionType(StrEnum):
    """The closed five-action vocabulary from :mod:`idea.tex`."""

    RETAIN_COMPRESS = "retain-compress"
    REFINE = "refine"
    SPLIT = "split"
    PRUNE = "prune"
    GENERATE = "generate"


class SplitBranch(StrEnum):
    """The unique child receiving one source task family after Split."""

    LOW = "low"
    HIGH = "high"


def _object(
    value: object,
    *,
    label: str,
    fields: frozenset[str],
) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict):
        raise ValueError(f"{label} must be a JSON object")
    if set(normalized) != fields:
        raise ValueError(f"{label} has incompatible fields")
    return normalized


def _text(value: JsonValue, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    require_non_empty_text(value, field=field)
    return value


def _number(value: JsonValue, *, field: str) -> float:
    return require_finite_number(value, field=field)


def _boolean(value: JsonValue, *, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be a boolean")
    return value


def _array(value: JsonValue, *, field: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return value


def _text_tuple(
    values: tuple[str, ...],
    *,
    field: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise ValueError(f"{field} must be a tuple")
    if not allow_empty and not values:
        raise ValueError(f"{field} cannot be empty")
    for value in values:
        require_non_empty_text(value, field=field)
    if len(set(values)) != len(values):
        raise ValueError(f"{field} must contain unique values")
    return values


def _wire_text_tuple(
    value: JsonValue,
    *,
    field: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    return _text_tuple(
        tuple(_text(item, field=field) for item in _array(value, field=field)),
        field=field,
        allow_empty=allow_empty,
    )


def _flow(value: float, quantile: float) -> tuple[float, float]:
    normalized_flow = require_finite_number(value, field="log_skill_marginal_flow")
    normalized_quantile = require_finite_number(quantile, field="flow_quantile")
    if not 0.0 <= normalized_quantile <= 1.0:
        raise ValueError("flow_quantile must lie in [0, 1]")
    return normalized_flow, normalized_quantile


def _confidence(value: float, k: float, *, field: str) -> tuple[float, float]:
    normalized_value = require_finite_number(value, field=field)
    normalized_k = require_finite_number(k, field="k")
    if normalized_k < 0.0:
        raise ValueError("k cannot be negative")
    return normalized_value, normalized_k


@dataclass(frozen=True, slots=True)
class PosteriorTaskFamilyMode:
    """One task-family-conditioned posterior mode supporting Split.

    The Bayesian Context coordinate is fixed by the extractor to
    ``TrajectoryRecord.task_family``. It is not a retrieval ``context_id``.
    """

    task_family: str
    cell_keys: tuple[str, ...]
    posterior_event_ids: tuple[str, ...]
    evidence_mass: float
    mean: float
    sigma: float
    lcb: float
    ucb: float
    within_cell_mean_span: float

    def __post_init__(self) -> None:
        require_non_empty_text(self.task_family, field="task_family")
        _text_tuple(self.cell_keys, field="cell_keys", allow_empty=False)
        _text_tuple(self.posterior_event_ids, field="posterior_event_ids", allow_empty=False)
        evidence_mass = require_finite_number(self.evidence_mass, field="evidence_mass")
        mean = require_finite_number(self.mean, field="mean")
        sigma = require_finite_number(self.sigma, field="sigma")
        lcb = require_finite_number(self.lcb, field="lcb")
        ucb = require_finite_number(self.ucb, field="ucb")
        span = require_finite_number(self.within_cell_mean_span, field="within_cell_mean_span")
        if evidence_mass < 0.0 or sigma < 0.0 or span < 0.0:
            raise ValueError("posterior mode mass, sigma, and span must be non-negative")
        if not 0.0 <= mean <= 1.0:
            raise ValueError("posterior mode mean must lie in [0, 1]")
        if not lcb <= mean <= ucb:
            raise ValueError("posterior mode interval must contain its mean")
        object.__setattr__(self, "evidence_mass", evidence_mass)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "lcb", lcb)
        object.__setattr__(self, "ucb", ucb)
        object.__setattr__(self, "within_cell_mean_span", span)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "cell_keys": list(self.cell_keys),
            "evidence_mass": self.evidence_mass,
            "lcb": self.lcb,
            "mean": self.mean,
            "posterior_event_ids": list(self.posterior_event_ids),
            "sigma": self.sigma,
            "task_family": self.task_family,
            "ucb": self.ucb,
            "within_cell_mean_span": self.within_cell_mean_span,
        }

    @classmethod
    def from_value(cls, value: object) -> PosteriorTaskFamilyMode:
        data = _object(
            value,
            label="PosteriorTaskFamilyMode",
            fields=frozenset(
                {
                    "cell_keys",
                    "evidence_mass",
                    "lcb",
                    "mean",
                    "posterior_event_ids",
                    "sigma",
                    "task_family",
                    "ucb",
                    "within_cell_mean_span",
                }
            ),
        )
        return cls(
            task_family=_text(data["task_family"], field="task_family"),
            cell_keys=_wire_text_tuple(data["cell_keys"], field="cell_keys", allow_empty=False),
            posterior_event_ids=_wire_text_tuple(
                data["posterior_event_ids"], field="posterior_event_ids", allow_empty=False
            ),
            evidence_mass=_number(data["evidence_mass"], field="evidence_mass"),
            mean=_number(data["mean"], field="mean"),
            sigma=_number(data["sigma"], field="sigma"),
            lcb=_number(data["lcb"], field="lcb"),
            ucb=_number(data["ucb"], field="ucb"),
            within_cell_mean_span=_number(
                data["within_cell_mean_span"], field="within_cell_mean_span"
            ),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class SplitTaskFamilyAssignment:
    """Bind one fully supported source task family to exactly one child."""

    mode: PosteriorTaskFamilyMode
    branch: SplitBranch

    def __post_init__(self) -> None:
        if not isinstance(self.mode, PosteriorTaskFamilyMode):
            raise TypeError("Split assignment requires PosteriorTaskFamilyMode")
        if not isinstance(self.branch, SplitBranch):
            raise TypeError("Split assignment requires SplitBranch")

    def to_value(self) -> dict[str, JsonValue]:
        return {"branch": self.branch.value, "mode": self.mode.to_value()}

    @classmethod
    def from_value(cls, value: object) -> SplitTaskFamilyAssignment:
        data = _object(
            value,
            label="SplitTaskFamilyAssignment",
            fields=frozenset({"branch", "mode"}),
        )
        return cls(
            mode=PosteriorTaskFamilyMode.from_value(data["mode"]),
            branch=SplitBranch(_text(data["branch"], field="branch")),
        )


@dataclass(frozen=True, slots=True)
class SplitModalityEvidence:
    """A complete, supported partition of the source task-family domain."""

    low_mode: PosteriorTaskFamilyMode
    high_mode: PosteriorTaskFamilyMode
    between_mean_gap: float
    intervals_disjoint: bool
    source_task_families: tuple[str, ...]
    assignments: tuple[SplitTaskFamilyAssignment, ...]
    separation_cutpoint: float

    def __post_init__(self) -> None:
        if not isinstance(self.low_mode, PosteriorTaskFamilyMode) or not isinstance(
            self.high_mode,
            PosteriorTaskFamilyMode,
        ):
            raise ValueError("split modality requires two posterior task-family modes")
        if self.low_mode.task_family == self.high_mode.task_family:
            raise ValueError("split modes must represent different task families")
        if set(self.low_mode.posterior_event_ids) & set(self.high_mode.posterior_event_ids):
            raise ValueError("split modes share posterior evidence")
        gap = require_finite_number(self.between_mean_gap, field="between_mean_gap")
        if not math.isclose(
            gap,
            self.high_mode.mean - self.low_mode.mean,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("split modality gap does not match mode means")
        if type(self.intervals_disjoint) is not bool or not self.intervals_disjoint:
            raise ValueError("split modality must record disjoint intervals")
        if not self.low_mode.ucb < self.high_mode.lcb:
            raise ValueError("split modes require disjoint credible intervals")
        if tuple(sorted(set(self.source_task_families))) != self.source_task_families:
            raise ValueError("source_task_families must be sorted and unique")
        if not self.source_task_families or "*" in self.source_task_families:
            raise ValueError("Split requires an expanded finite source domain")
        if not isinstance(self.assignments, tuple) or any(
            not isinstance(item, SplitTaskFamilyAssignment) for item in self.assignments
        ):
            raise TypeError("Split assignments must contain SplitTaskFamilyAssignment values")
        assigned = tuple(item.mode.task_family for item in self.assignments)
        if tuple(sorted(assigned)) != self.source_task_families or len(set(assigned)) != len(
            assigned
        ):
            raise ValueError("Split assignments must cover the source domain exactly once")
        if not self.low_task_families or not self.high_task_families:
            raise ValueError("both Split children must be non-empty")
        if set(self.low_task_families) & set(self.high_task_families):
            raise ValueError("Split children must be disjoint")
        if set(self.low_task_families) | set(self.high_task_families) != set(
            self.source_task_families
        ):
            raise ValueError("Split children must exhaust the source domain")
        if self.low_mode.task_family not in self.low_task_families:
            raise ValueError("low anchor is assigned to the wrong child")
        if self.high_mode.task_family not in self.high_task_families:
            raise ValueError("high anchor is assigned to the wrong child")
        cutpoint = require_finite_number(self.separation_cutpoint, field="separation_cutpoint")
        if not self.low_mode.ucb < cutpoint < self.high_mode.lcb:
            raise ValueError("Split separation cutpoint must lie between anchor intervals")
        for assignment in self.assignments:
            if assignment.branch is SplitBranch.LOW and not assignment.mode.ucb < cutpoint:
                raise ValueError("low Split assignment crosses the separation cutpoint")
            if assignment.branch is SplitBranch.HIGH and not assignment.mode.lcb > cutpoint:
                raise ValueError("high Split assignment crosses the separation cutpoint")
        object.__setattr__(self, "between_mean_gap", gap)
        object.__setattr__(self, "separation_cutpoint", cutpoint)

    @property
    def low_task_families(self) -> tuple[str, ...]:
        return tuple(
            item.mode.task_family for item in self.assignments if item.branch is SplitBranch.LOW
        )

    @property
    def high_task_families(self) -> tuple[str, ...]:
        return tuple(
            item.mode.task_family for item in self.assignments if item.branch is SplitBranch.HIGH
        )

    @property
    def posterior_event_ids(self) -> tuple[str, ...]:
        return tuple(
            event_id
            for assignment in self.assignments
            for event_id in assignment.mode.posterior_event_ids
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "between_mean_gap": self.between_mean_gap,
            "assignments": [item.to_value() for item in self.assignments],
            "high_mode": self.high_mode.to_value(),
            "intervals_disjoint": self.intervals_disjoint,
            "low_mode": self.low_mode.to_value(),
            "separation_cutpoint": self.separation_cutpoint,
            "source_task_families": list(self.source_task_families),
        }

    @classmethod
    def from_value(cls, value: object) -> SplitModalityEvidence:
        data = _object(
            value,
            label="SplitModalityEvidence",
            fields=frozenset(
                {
                    "between_mean_gap",
                    "assignments",
                    "high_mode",
                    "intervals_disjoint",
                    "low_mode",
                    "separation_cutpoint",
                    "source_task_families",
                }
            ),
        )
        return cls(
            low_mode=PosteriorTaskFamilyMode.from_value(data["low_mode"]),
            high_mode=PosteriorTaskFamilyMode.from_value(data["high_mode"]),
            between_mean_gap=_number(data["between_mean_gap"], field="between_mean_gap"),
            intervals_disjoint=_boolean(
                data["intervals_disjoint"],
                field="intervals_disjoint",
            ),
            source_task_families=_wire_text_tuple(
                data["source_task_families"], field="source_task_families", allow_empty=False
            ),
            assignments=tuple(
                SplitTaskFamilyAssignment.from_value(item)
                for item in _array(data["assignments"], field="assignments")
            ),
            separation_cutpoint=_number(data["separation_cutpoint"], field="separation_cutpoint"),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class RetainEvidence:
    log_skill_marginal_flow: float
    flow_quantile: float
    lcb: float
    k: float
    posterior_event_ids: tuple[str, ...] = ()

    evidence_type: ClassVar[str] = "retain"

    def __post_init__(self) -> None:
        flow, quantile = _flow(self.log_skill_marginal_flow, self.flow_quantile)
        lcb, k = _confidence(self.lcb, self.k, field="lcb")
        _text_tuple(
            self.posterior_event_ids,
            field="posterior_event_ids",
            allow_empty=True,
        )
        object.__setattr__(self, "log_skill_marginal_flow", flow)
        object.__setattr__(self, "flow_quantile", quantile)
        object.__setattr__(self, "lcb", lcb)
        object.__setattr__(self, "k", k)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "evidence_type": self.evidence_type,
            "flow_quantile": self.flow_quantile,
            "k": self.k,
            "lcb": self.lcb,
            "log_skill_marginal_flow": self.log_skill_marginal_flow,
            "posterior_event_ids": list(self.posterior_event_ids),
        }

    @classmethod
    def from_value(cls, value: object) -> RetainEvidence:
        data = _object(
            value,
            label="RetainEvidence",
            fields=frozenset(
                {
                    "evidence_type",
                    "flow_quantile",
                    "k",
                    "lcb",
                    "log_skill_marginal_flow",
                    "posterior_event_ids",
                }
            ),
        )
        if data["evidence_type"] != cls.evidence_type:
            raise ValueError("RetainEvidence has an incompatible evidence_type")
        return cls(
            log_skill_marginal_flow=_number(
                data["log_skill_marginal_flow"],
                field="log_skill_marginal_flow",
            ),
            flow_quantile=_number(data["flow_quantile"], field="flow_quantile"),
            lcb=_number(data["lcb"], field="lcb"),
            k=_number(data["k"], field="k"),
            posterior_event_ids=_wire_text_tuple(
                data["posterior_event_ids"],
                field="posterior_event_ids",
                allow_empty=True,
            ),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class RefineEvidence:
    log_skill_marginal_flow: float
    flow_quantile: float
    lcb: float
    k: float
    posterior_event_ids: tuple[str, ...]
    target_context_keys: tuple[str, ...]
    target_contexts: tuple[ContextFeature, ...] = ()

    evidence_type: ClassVar[str] = "refine"

    def __post_init__(self) -> None:
        flow, quantile = _flow(self.log_skill_marginal_flow, self.flow_quantile)
        lcb, k = _confidence(self.lcb, self.k, field="lcb")
        _text_tuple(
            self.posterior_event_ids,
            field="posterior_event_ids",
            allow_empty=False,
        )
        _text_tuple(
            self.target_context_keys,
            field="target_context_keys",
            allow_empty=False,
        )
        if not isinstance(self.target_contexts, tuple):
            raise ValueError("Refine context coordinates must be immutable")
        if self.target_contexts and (
            len(self.target_contexts) != len(self.target_context_keys)
            or any(not isinstance(z, ContextFeature) for z in self.target_contexts)
        ):
            raise ValueError("Refine context coordinates differ from target keys")
        object.__setattr__(self, "log_skill_marginal_flow", flow)
        object.__setattr__(self, "flow_quantile", quantile)
        object.__setattr__(self, "lcb", lcb)
        object.__setattr__(self, "k", k)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "evidence_type": self.evidence_type,
            "flow_quantile": self.flow_quantile,
            "k": self.k,
            "lcb": self.lcb,
            "log_skill_marginal_flow": self.log_skill_marginal_flow,
            "posterior_event_ids": list(self.posterior_event_ids),
            "target_context_keys": list(self.target_context_keys),
            "target_contexts": [z.to_value() for z in self.target_contexts],
        }

    @classmethod
    def from_value(cls, value: object) -> RefineEvidence:
        # Historical decisions remain readable, but cannot be authored without
        # their actual post-hoc coordinates; do not invent missing evidence.
        if isinstance(value, dict) and "target_contexts" not in value:
            value = {**value, "target_contexts": []}
        data = _object(
            value,
            label="RefineEvidence",
            fields=frozenset(
                {
                    "evidence_type",
                    "flow_quantile",
                    "k",
                    "lcb",
                    "log_skill_marginal_flow",
                    "posterior_event_ids",
                    "target_context_keys",
                    "target_contexts",
                }
            ),
        )
        if data["evidence_type"] != cls.evidence_type:
            raise ValueError("RefineEvidence has an incompatible evidence_type")
        return cls(
            log_skill_marginal_flow=_number(
                data["log_skill_marginal_flow"],
                field="log_skill_marginal_flow",
            ),
            flow_quantile=_number(data["flow_quantile"], field="flow_quantile"),
            lcb=_number(data["lcb"], field="lcb"),
            k=_number(data["k"], field="k"),
            posterior_event_ids=_wire_text_tuple(
                data["posterior_event_ids"],
                field="posterior_event_ids",
                allow_empty=False,
            ),
            target_contexts=tuple(
                ContextFeature.from_value(z)
                for z in _array(data["target_contexts"], field="target_contexts")
            ),
            target_context_keys=_wire_text_tuple(
                data["target_context_keys"],
                field="target_context_keys",
                allow_empty=False,
            ),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class SplitEvidence:
    log_skill_marginal_flow: float
    flow_quantile: float
    modality: SplitModalityEvidence

    evidence_type: ClassVar[str] = "split"

    def __post_init__(self) -> None:
        flow, quantile = _flow(self.log_skill_marginal_flow, self.flow_quantile)
        if not isinstance(self.modality, SplitModalityEvidence):
            raise ValueError("modality must be SplitModalityEvidence")
        object.__setattr__(self, "log_skill_marginal_flow", flow)
        object.__setattr__(self, "flow_quantile", quantile)

    @property
    def posterior_event_ids(self) -> tuple[str, ...]:
        return self.modality.posterior_event_ids

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "evidence_type": self.evidence_type,
            "flow_quantile": self.flow_quantile,
            "log_skill_marginal_flow": self.log_skill_marginal_flow,
            "modality": self.modality.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> SplitEvidence:
        data = _object(
            value,
            label="SplitEvidence",
            fields=frozenset(
                {
                    "evidence_type",
                    "flow_quantile",
                    "log_skill_marginal_flow",
                    "modality",
                }
            ),
        )
        if data["evidence_type"] != cls.evidence_type:
            raise ValueError("SplitEvidence has an incompatible evidence_type")
        return cls(
            log_skill_marginal_flow=_number(
                data["log_skill_marginal_flow"],
                field="log_skill_marginal_flow",
            ),
            flow_quantile=_number(data["flow_quantile"], field="flow_quantile"),
            modality=SplitModalityEvidence.from_value(data["modality"]),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class PruneEvidence:
    flow_evidence_kind: str
    log_skill_marginal_flow: float | None
    flow_quantile: float | None
    zero_invocation_window_id: str | None
    ucb: float
    k: float
    posterior_event_ids: tuple[str, ...]

    evidence_type: ClassVar[str] = "prune"

    def __post_init__(self) -> None:
        if self.flow_evidence_kind == "observed-low":
            if (
                self.log_skill_marginal_flow is None
                or self.flow_quantile is None
                or self.zero_invocation_window_id is not None
            ):
                raise ValueError("observed-low Prune requires only observed flow fields")
            flow, quantile = _flow(
                self.log_skill_marginal_flow,
                self.flow_quantile,
            )
        elif self.flow_evidence_kind == "zero-invocation":
            if (
                self.log_skill_marginal_flow is not None
                or self.flow_quantile is not None
                or self.zero_invocation_window_id is None
            ):
                raise ValueError("zero-invocation Prune requires only its window identity")
            require_non_empty_text(
                self.zero_invocation_window_id,
                field="zero_invocation_window_id",
            )
            flow = None
            quantile = None
        else:
            raise ValueError("unsupported Prune flow evidence kind")
        ucb, k = _confidence(self.ucb, self.k, field="ucb")
        _text_tuple(
            self.posterior_event_ids,
            field="posterior_event_ids",
            allow_empty=False,
        )
        object.__setattr__(self, "log_skill_marginal_flow", flow)
        object.__setattr__(self, "flow_quantile", quantile)
        object.__setattr__(self, "ucb", ucb)
        object.__setattr__(self, "k", k)

    @classmethod
    def observed_low(
        cls,
        *,
        log_skill_marginal_flow: float,
        flow_quantile: float,
        ucb: float,
        k: float,
        posterior_event_ids: tuple[str, ...],
    ) -> PruneEvidence:
        return cls(
            flow_evidence_kind="observed-low",
            log_skill_marginal_flow=log_skill_marginal_flow,
            flow_quantile=flow_quantile,
            zero_invocation_window_id=None,
            ucb=ucb,
            k=k,
            posterior_event_ids=posterior_event_ids,
        )

    @classmethod
    def zero_invocation(
        cls,
        *,
        window_id: str,
        ucb: float,
        k: float,
        posterior_event_ids: tuple[str, ...],
    ) -> PruneEvidence:
        return cls(
            flow_evidence_kind="zero-invocation",
            log_skill_marginal_flow=None,
            flow_quantile=None,
            zero_invocation_window_id=window_id,
            ucb=ucb,
            k=k,
            posterior_event_ids=posterior_event_ids,
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "evidence_type": self.evidence_type,
            "flow_evidence_kind": self.flow_evidence_kind,
            "flow_quantile": self.flow_quantile,
            "k": self.k,
            "log_skill_marginal_flow": self.log_skill_marginal_flow,
            "posterior_event_ids": list(self.posterior_event_ids),
            "ucb": self.ucb,
            "zero_invocation_window_id": self.zero_invocation_window_id,
        }

    @classmethod
    def from_value(cls, value: object) -> PruneEvidence:
        data = _object(
            value,
            label="PruneEvidence",
            fields=frozenset(
                {
                    "evidence_type",
                    "flow_evidence_kind",
                    "flow_quantile",
                    "k",
                    "log_skill_marginal_flow",
                    "posterior_event_ids",
                    "ucb",
                    "zero_invocation_window_id",
                }
            ),
        )
        if data["evidence_type"] != cls.evidence_type:
            raise ValueError("PruneEvidence has an incompatible evidence_type")
        flow = data["log_skill_marginal_flow"]
        quantile = data["flow_quantile"]
        window_id = data["zero_invocation_window_id"]
        if flow is not None and (isinstance(flow, bool) or not isinstance(flow, int | float)):
            raise TypeError("log_skill_marginal_flow must be numeric or null")
        if quantile is not None and (
            isinstance(quantile, bool) or not isinstance(quantile, int | float)
        ):
            raise TypeError("flow_quantile must be numeric or null")
        if window_id is not None and not isinstance(window_id, str):
            raise TypeError("zero_invocation_window_id must be text or null")
        return cls(
            flow_evidence_kind=_text(data["flow_evidence_kind"], field="flow_evidence_kind"),
            log_skill_marginal_flow=None if flow is None else float(flow),
            flow_quantile=None if quantile is None else float(quantile),
            zero_invocation_window_id=window_id,
            ucb=_number(data["ucb"], field="ucb"),
            k=_number(data["k"], field="k"),
            posterior_event_ids=_wire_text_tuple(
                data["posterior_event_ids"],
                field="posterior_event_ids",
                allow_empty=False,
            ),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class GenerateEvidence:
    importance_edge_ids: tuple[str, ...]
    minimum_absolute_log_importance: float
    importance_quantile: float
    importance_semantics: str

    evidence_type: ClassVar[str] = "generate"

    def __post_init__(self) -> None:
        _text_tuple(
            self.importance_edge_ids,
            field="importance_edge_ids",
            allow_empty=False,
        )
        minimum = require_finite_number(
            self.minimum_absolute_log_importance,
            field="minimum_absolute_log_importance",
        )
        quantile = require_finite_number(self.importance_quantile, field="importance_quantile")
        if minimum <= 0.0:
            raise ValueError("Generate evidence requires a positive absolute floor")
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("Generate evidence quantile must lie in [0, 1]")
        if self.importance_semantics != "absolute-log-density-ratio@1":
            raise ValueError("Generate evidence uses unsupported semantics")
        object.__setattr__(self, "minimum_absolute_log_importance", minimum)
        object.__setattr__(self, "importance_quantile", quantile)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "evidence_type": self.evidence_type,
            "importance_edge_ids": list(self.importance_edge_ids),
            "importance_quantile": self.importance_quantile,
            "importance_semantics": self.importance_semantics,
            "minimum_absolute_log_importance": self.minimum_absolute_log_importance,
        }

    @classmethod
    def from_value(cls, value: object) -> GenerateEvidence:
        data = _object(
            value,
            label="GenerateEvidence",
            fields=frozenset(
                {
                    "evidence_type",
                    "importance_edge_ids",
                    "importance_quantile",
                    "importance_semantics",
                    "minimum_absolute_log_importance",
                }
            ),
        )
        if data["evidence_type"] != cls.evidence_type:
            raise ValueError("GenerateEvidence has an incompatible evidence_type")
        return cls(
            importance_edge_ids=_wire_text_tuple(
                data["importance_edge_ids"],
                field="importance_edge_ids",
                allow_empty=False,
            ),
            minimum_absolute_log_importance=_number(
                data["minimum_absolute_log_importance"],
                field="minimum_absolute_log_importance",
            ),
            importance_quantile=_number(data["importance_quantile"], field="importance_quantile"),
            importance_semantics=_text(data["importance_semantics"], field="importance_semantics"),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


EvolutionEvidence: TypeAlias = (
    RetainEvidence | RefineEvidence | SplitEvidence | PruneEvidence | GenerateEvidence
)


def evolution_evidence_from_value(value: object) -> EvolutionEvidence:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict):
        raise ValueError("evolution evidence must be a JSON object")
    evidence_type = normalized.get("evidence_type")
    if evidence_type == RetainEvidence.evidence_type:
        return RetainEvidence.from_value(normalized)
    if evidence_type == RefineEvidence.evidence_type:
        return RefineEvidence.from_value(normalized)
    if evidence_type == SplitEvidence.evidence_type:
        return SplitEvidence.from_value(normalized)
    if evidence_type == PruneEvidence.evidence_type:
        return PruneEvidence.from_value(normalized)
    if evidence_type == GenerateEvidence.evidence_type:
        return GenerateEvidence.from_value(normalized)
    raise ValueError("unsupported evolution evidence type")


def _common_action_validation(
    *,
    action_id: str,
    proposal_content_hash: str,
    phase_event_id: str,
    library_version_before: str,
    library_version_after: str,
    lineage_ref: str,
    rationale_text: str,
    wire_format: str,
) -> None:
    for field, value in (
        ("action_id", action_id),
        ("phase_event_id", phase_event_id),
        ("library_version_before", library_version_before),
        ("library_version_after", library_version_after),
        ("lineage_ref", lineage_ref),
        ("rationale_text", rationale_text),
    ):
        require_non_empty_text(value, field=field)
    if library_version_before == library_version_after:
        raise ValueError("evolution action must create a distinct library version")
    validate_sha256(proposal_content_hash)
    if wire_format != EVOLUTION_FORMAT:
        raise ValueError(f"format must be {EVOLUTION_FORMAT!r}")


_COMMON_ACTION_FIELDS = frozenset(
    {
        "action_id",
        "action_type",
        "evidence",
        "format",
        "library_version_after",
        "library_version_before",
        "lineage_ref",
        "phase_event_id",
        "proposal_content_hash",
        "rationale_text",
    }
)


def _common_action_value(record: EvolutionActionRecord) -> dict[str, JsonValue]:
    return {
        "action_id": record.action_id,
        "action_type": record.action_type.value,
        "evidence": record.evidence.to_value(),
        "format": record.format,
        "library_version_after": record.library_version_after,
        "library_version_before": record.library_version_before,
        "lineage_ref": record.lineage_ref,
        "phase_event_id": record.phase_event_id,
        "proposal_content_hash": record.proposal_content_hash,
        "rationale_text": record.rationale_text,
    }


def _common_action_kwargs(data: dict[str, JsonValue]) -> dict[str, str]:
    if data["format"] != EVOLUTION_FORMAT:
        raise ValueError("evolution action has an incompatible format")
    return {
        "action_id": _text(data["action_id"], field="action_id"),
        "proposal_content_hash": _text(
            data["proposal_content_hash"], field="proposal_content_hash"
        ),
        "phase_event_id": _text(data["phase_event_id"], field="phase_event_id"),
        "library_version_before": _text(
            data["library_version_before"],
            field="library_version_before",
        ),
        "library_version_after": _text(
            data["library_version_after"],
            field="library_version_after",
        ),
        "lineage_ref": _text(data["lineage_ref"], field="lineage_ref"),
        "rationale_text": _text(data["rationale_text"], field="rationale_text"),
    }


@dataclass(frozen=True, slots=True)
class RetainCompressActionRecord:
    action_id: str
    proposal_content_hash: str
    phase_event_id: str
    library_version_before: str
    library_version_after: str
    target_skill_id: str
    produced_skill_id: str
    lineage_ref: str
    evidence: RetainEvidence
    rationale_text: str
    format: str = EVOLUTION_FORMAT

    action_type: ClassVar[EvolutionActionType] = EvolutionActionType.RETAIN_COMPRESS

    def __post_init__(self) -> None:
        _common_action_validation(
            action_id=self.action_id,
            proposal_content_hash=self.proposal_content_hash,
            phase_event_id=self.phase_event_id,
            library_version_before=self.library_version_before,
            library_version_after=self.library_version_after,
            lineage_ref=self.lineage_ref,
            rationale_text=self.rationale_text,
            wire_format=self.format,
        )
        require_non_empty_text(self.target_skill_id, field="target_skill_id")
        require_non_empty_text(self.produced_skill_id, field="produced_skill_id")
        if not isinstance(self.evidence, RetainEvidence):
            raise TypeError("full retain-compress requires RetainEvidence")

    @property
    def target_skill_ids(self) -> tuple[str, ...]:
        return (self.target_skill_id,)

    @property
    def produced_skill_ids(self) -> tuple[str, ...]:
        return (self.produced_skill_id,)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            **_common_action_value(self),
            "produced_skill_id": self.produced_skill_id,
            "target_skill_id": self.target_skill_id,
        }

    @classmethod
    def from_value(cls, value: object) -> RetainCompressActionRecord:
        data = _action_object(
            value,
            action=cls.action_type,
            extra_fields=frozenset({"produced_skill_id", "target_skill_id"}),
        )
        evidence = evolution_evidence_from_value(data["evidence"])
        if not isinstance(evidence, RetainEvidence):
            raise ValueError("retain-compress wire evidence must be RetainEvidence")
        return cls(
            **_common_action_kwargs(data),
            target_skill_id=_text(data["target_skill_id"], field="target_skill_id"),
            produced_skill_id=_text(data["produced_skill_id"], field="produced_skill_id"),
            evidence=evidence,
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class RefineActionRecord:
    action_id: str
    proposal_content_hash: str
    phase_event_id: str
    library_version_before: str
    library_version_after: str
    target_skill_id: str
    produced_skill_id: str
    lineage_ref: str
    evidence: RefineEvidence
    rationale_text: str
    format: str = EVOLUTION_FORMAT

    action_type: ClassVar[EvolutionActionType] = EvolutionActionType.REFINE

    def __post_init__(self) -> None:
        _common_action_validation(
            action_id=self.action_id,
            proposal_content_hash=self.proposal_content_hash,
            phase_event_id=self.phase_event_id,
            library_version_before=self.library_version_before,
            library_version_after=self.library_version_after,
            lineage_ref=self.lineage_ref,
            rationale_text=self.rationale_text,
            wire_format=self.format,
        )
        require_non_empty_text(self.target_skill_id, field="target_skill_id")
        require_non_empty_text(self.produced_skill_id, field="produced_skill_id")
        if not isinstance(self.evidence, RefineEvidence):
            raise ValueError("refine requires RefineEvidence")

    @property
    def target_skill_ids(self) -> tuple[str, ...]:
        return (self.target_skill_id,)

    @property
    def produced_skill_ids(self) -> tuple[str, ...]:
        return (self.produced_skill_id,)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            **_common_action_value(self),
            "produced_skill_id": self.produced_skill_id,
            "target_skill_id": self.target_skill_id,
        }

    @classmethod
    def from_value(cls, value: object) -> RefineActionRecord:
        data = _action_object(
            value,
            action=cls.action_type,
            extra_fields=frozenset({"produced_skill_id", "target_skill_id"}),
        )
        evidence = evolution_evidence_from_value(data["evidence"])
        if not isinstance(evidence, RefineEvidence):
            raise ValueError("refine wire evidence has the wrong type")
        return cls(
            **_common_action_kwargs(data),
            target_skill_id=_text(data["target_skill_id"], field="target_skill_id"),
            produced_skill_id=_text(data["produced_skill_id"], field="produced_skill_id"),
            evidence=evidence,
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class SplitActionRecord:
    action_id: str
    proposal_content_hash: str
    phase_event_id: str
    library_version_before: str
    library_version_after: str
    target_skill_id: str
    produced_skill_ids: tuple[str, str]
    lineage_ref: str
    evidence: SplitEvidence
    rationale_text: str
    format: str = EVOLUTION_FORMAT

    action_type: ClassVar[EvolutionActionType] = EvolutionActionType.SPLIT

    def __post_init__(self) -> None:
        _common_action_validation(
            action_id=self.action_id,
            proposal_content_hash=self.proposal_content_hash,
            phase_event_id=self.phase_event_id,
            library_version_before=self.library_version_before,
            library_version_after=self.library_version_after,
            lineage_ref=self.lineage_ref,
            rationale_text=self.rationale_text,
            wire_format=self.format,
        )
        require_non_empty_text(self.target_skill_id, field="target_skill_id")
        _text_tuple(
            self.produced_skill_ids,
            field="produced_skill_ids",
            allow_empty=False,
        )
        if len(self.produced_skill_ids) != 2:
            raise ValueError("split requires exactly two produced skills")
        if not isinstance(self.evidence, SplitEvidence):
            raise ValueError("split requires SplitEvidence")

    @property
    def target_skill_ids(self) -> tuple[str, ...]:
        return (self.target_skill_id,)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            **_common_action_value(self),
            "produced_skill_ids": list(self.produced_skill_ids),
            "target_skill_id": self.target_skill_id,
        }

    @classmethod
    def from_value(cls, value: object) -> SplitActionRecord:
        data = _action_object(
            value,
            action=cls.action_type,
            extra_fields=frozenset({"produced_skill_ids", "target_skill_id"}),
        )
        evidence = evolution_evidence_from_value(data["evidence"])
        if not isinstance(evidence, SplitEvidence):
            raise ValueError("split wire evidence has the wrong type")
        products = _wire_text_tuple(
            data["produced_skill_ids"],
            field="produced_skill_ids",
            allow_empty=False,
        )
        if len(products) != 2:
            raise ValueError("split wire record requires exactly two products")
        return cls(
            **_common_action_kwargs(data),
            target_skill_id=_text(data["target_skill_id"], field="target_skill_id"),
            produced_skill_ids=(products[0], products[1]),
            evidence=evidence,
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class PruneActionRecord:
    action_id: str
    proposal_content_hash: str
    phase_event_id: str
    library_version_before: str
    library_version_after: str
    target_skill_id: str
    lineage_ref: str
    evidence: PruneEvidence
    rationale_text: str
    format: str = EVOLUTION_FORMAT

    action_type: ClassVar[EvolutionActionType] = EvolutionActionType.PRUNE

    def __post_init__(self) -> None:
        _common_action_validation(
            action_id=self.action_id,
            proposal_content_hash=self.proposal_content_hash,
            phase_event_id=self.phase_event_id,
            library_version_before=self.library_version_before,
            library_version_after=self.library_version_after,
            lineage_ref=self.lineage_ref,
            rationale_text=self.rationale_text,
            wire_format=self.format,
        )
        require_non_empty_text(self.target_skill_id, field="target_skill_id")
        if not isinstance(self.evidence, PruneEvidence):
            raise ValueError("prune requires PruneEvidence")

    @property
    def target_skill_ids(self) -> tuple[str, ...]:
        return (self.target_skill_id,)

    @property
    def produced_skill_ids(self) -> tuple[str, ...]:
        return ()

    def to_value(self) -> dict[str, JsonValue]:
        return {
            **_common_action_value(self),
            "target_skill_id": self.target_skill_id,
        }

    @classmethod
    def from_value(cls, value: object) -> PruneActionRecord:
        data = _action_object(
            value,
            action=cls.action_type,
            extra_fields=frozenset({"target_skill_id"}),
        )
        evidence = evolution_evidence_from_value(data["evidence"])
        if not isinstance(evidence, PruneEvidence):
            raise ValueError("prune wire evidence has the wrong type")
        return cls(
            **_common_action_kwargs(data),
            target_skill_id=_text(data["target_skill_id"], field="target_skill_id"),
            evidence=evidence,
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class GenerateActionRecord:
    action_id: str
    proposal_content_hash: str
    phase_event_id: str
    library_version_before: str
    library_version_after: str
    produced_skill_id: str
    lineage_ref: str
    evidence: GenerateEvidence
    rationale_text: str
    format: str = EVOLUTION_FORMAT

    action_type: ClassVar[EvolutionActionType] = EvolutionActionType.GENERATE

    def __post_init__(self) -> None:
        _common_action_validation(
            action_id=self.action_id,
            proposal_content_hash=self.proposal_content_hash,
            phase_event_id=self.phase_event_id,
            library_version_before=self.library_version_before,
            library_version_after=self.library_version_after,
            lineage_ref=self.lineage_ref,
            rationale_text=self.rationale_text,
            wire_format=self.format,
        )
        require_non_empty_text(self.produced_skill_id, field="produced_skill_id")
        if not isinstance(self.evidence, GenerateEvidence):
            raise ValueError("generate requires GenerateEvidence")

    @property
    def target_skill_ids(self) -> tuple[str, ...]:
        return ()

    @property
    def produced_skill_ids(self) -> tuple[str, ...]:
        return (self.produced_skill_id,)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            **_common_action_value(self),
            "produced_skill_id": self.produced_skill_id,
        }

    @classmethod
    def from_value(cls, value: object) -> GenerateActionRecord:
        data = _action_object(
            value,
            action=cls.action_type,
            extra_fields=frozenset({"produced_skill_id"}),
        )
        evidence = evolution_evidence_from_value(data["evidence"])
        if not isinstance(evidence, GenerateEvidence):
            raise ValueError("generate wire evidence has the wrong type")
        return cls(
            **_common_action_kwargs(data),
            produced_skill_id=_text(data["produced_skill_id"], field="produced_skill_id"),
            evidence=evidence,
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


EvolutionActionRecord: TypeAlias = (
    RetainCompressActionRecord
    | RefineActionRecord
    | SplitActionRecord
    | PruneActionRecord
    | GenerateActionRecord
)


def _action_object(
    value: object,
    *,
    action: EvolutionActionType,
    extra_fields: frozenset[str],
) -> dict[str, JsonValue]:
    data = _object(
        value,
        label=f"{action.value} action",
        fields=_COMMON_ACTION_FIELDS | extra_fields,
    )
    if data["action_type"] != action.value:
        raise ValueError(f"action_type must be {action.value!r}")
    return data


def evolution_action_from_value(value: object) -> EvolutionActionRecord:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict):
        raise ValueError("evolution action must be a JSON object")
    raw_action = normalized.get("action_type")
    if not isinstance(raw_action, str):
        raise ValueError("evolution action_type must be text")
    try:
        action = EvolutionActionType(raw_action)
    except ValueError as error:
        raise ValueError("unsupported evolution action_type") from error
    if action is EvolutionActionType.RETAIN_COMPRESS:
        return RetainCompressActionRecord.from_value(normalized)
    if action is EvolutionActionType.REFINE:
        return RefineActionRecord.from_value(normalized)
    if action is EvolutionActionType.SPLIT:
        return SplitActionRecord.from_value(normalized)
    if action is EvolutionActionType.PRUNE:
        return PruneActionRecord.from_value(normalized)
    return GenerateActionRecord.from_value(normalized)


__all__ = [
    "EVOLUTION_FORMAT",
    "EvolutionActionRecord",
    "EvolutionActionType",
    "EvolutionEvidence",
    "GenerateActionRecord",
    "GenerateEvidence",
    "PosteriorTaskFamilyMode",
    "PruneActionRecord",
    "PruneEvidence",
    "RefineActionRecord",
    "RefineEvidence",
    "RetainCompressActionRecord",
    "RetainEvidence",
    "SplitActionRecord",
    "SplitBranch",
    "SplitEvidence",
    "SplitModalityEvidence",
    "SplitTaskFamilyAssignment",
    "evolution_action_from_value",
    "evolution_evidence_from_value",
]
