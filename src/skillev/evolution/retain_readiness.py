"""Fixed public synthetic Retain development and untouched holdout suites.

These fixtures measure frozen-base branch reliability, not benchmark quality.
They contain no benchmark prompts, answers, rewards, or verifier material.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from skillev.contracts import (
    FailureMode,
    HorizonBucket,
    JsonValue,
    TokenBucket,
    stable_hash,
)
from skillev.runtime import (
    SkillApplicability,
    SkillDocument,
    SkillManifest,
    SkillRequirement,
)

from .authoring import AuthoringSamplingConfig, RetainAuthoringRequest
from .evidence import AuthoringActionKind, AuthoringEdgeEvidence

RETAIN_READINESS_SUITE_FORMAT: Final = "skillev-retain-readiness-suite@1"
RETAIN_DEVELOPMENT_SUITE_VERSION: Final = "retain-development@1"
RETAIN_HOLDOUT_SUITE_VERSION: Final = "retain-holdout@1"
RETAIN_READINESS_BASE_SEED: Final = 20_260_801
RETAIN_AUTHORING_SAMPLING: Final = AuthoringSamplingConfig(
    temperature=0.1,
    top_p=0.95,
)


class RetainSuiteKind(StrEnum):
    DEVELOPMENT = "development"
    HOLDOUT = "holdout"


@dataclass(frozen=True, slots=True)
class RetainReadinessCase:
    case_id: str
    profiles: tuple[str, ...]
    source: SkillDocument
    edge_exemplar: AuthoringEdgeEvidence
    evidence_summary: str
    seed: int

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("Retain readiness case ID cannot be empty")
        if tuple(sorted(set(self.profiles))) != self.profiles or not self.profiles:
            raise ValueError("Retain readiness profiles must be sorted, unique, and nonempty")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("Retain readiness seed must be an unsigned 64-bit integer")

    @property
    def request(self) -> RetainAuthoringRequest:
        return RetainAuthoringRequest(
            source=self.source,
            edge_exemplars=(self.edge_exemplar,),
            evidence_summary=self.evidence_summary,
            seed=self.seed,
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "case_id": self.case_id,
            "edge_exemplar": self.edge_exemplar.to_value(),
            "evidence_summary": self.evidence_summary,
            "profiles": list(self.profiles),
            "seed": self.seed,
            "source": self.source.to_value(),
        }


@dataclass(frozen=True, slots=True)
class RetainReadinessSuite:
    kind: RetainSuiteKind
    version: str
    cases: tuple[RetainReadinessCase, ...]
    format: str = RETAIN_READINESS_SUITE_FORMAT

    def __post_init__(self) -> None:
        if self.format != RETAIN_READINESS_SUITE_FORMAT:
            raise ValueError("Retain readiness suite format is unsupported")
        if not self.version.strip() or not self.cases:
            raise ValueError("Retain readiness suite requires a version and cases")
        ids = tuple(item.case_id for item in self.cases)
        if len(set(ids)) != len(ids):
            raise ValueError("Retain readiness case IDs must be unique")
        required_profiles = {
            "few-requirements",
            "many-requirements",
            "short-instructions",
            "medium-instructions",
            "long-instructions",
            "single-task-family",
            "multiple-task-families",
            "with-tool",
            "without-tool",
            "highly-repetitive",
            "structured-low-redundancy",
            "near-minimal-legal",
        }
        observed = {profile for item in self.cases for profile in item.profiles}
        if not required_profiles <= observed:
            raise ValueError("Retain readiness suite does not cover the fixed profile matrix")

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "cases": [item.to_value() for item in self.cases],
            "format": self.format,
            "kind": self.kind.value,
            "sampling": RETAIN_AUTHORING_SAMPLING.to_value(),
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class _CaseSpec:
    name: str
    profiles: tuple[str, ...]
    task_families: tuple[str, ...]
    context: str
    required_tools: tuple[str, ...]
    title: str
    summary: str
    instructions: str
    requirement_texts: tuple[str, ...]


def _case(
    spec: _CaseSpec,
    *,
    suite_version: str,
    ordinal: int,
) -> RetainReadinessCase:
    suite_identifier = suite_version.replace("@", "-")
    applicability = SkillApplicability(
        task_families=spec.task_families,
        contexts=(spec.context,),
        required_tools=spec.required_tools,
        excluded_contexts=(),
    )
    requirements = tuple(
        SkillRequirement(
            requirement_id=f"{spec.name}-requirement-{index}",
            text=text,
        )
        for index, text in enumerate(spec.requirement_texts, start=1)
    )
    content: dict[str, JsonValue] = {
        "applicability": applicability.to_value(),
        "instructions": spec.instructions,
        "requirements": [item.to_value() for item in requirements],
        "summary": spec.summary,
        "title": spec.title,
    }
    source = SkillDocument(
        manifest=SkillManifest(
            skill_id=f"skill-{suite_identifier}-{spec.name}",
            version="1",
            content_hash=stable_hash(content),
            input_schema_id="retain-readiness-input@1",
            output_schema_id="retain-readiness-output@1",
            license_id="project-owned-public-synthetic",
            provenance_hash=stable_hash({"case": spec.name, "suite_version": suite_version}),
        ),
        title=spec.title,
        summary=spec.summary,
        instructions=spec.instructions,
        applicability=applicability,
        requirements=requirements,
    )
    edge = AuthoringEdgeEvidence(
        edge_id=f"{suite_version}-{spec.name}:1",
        task_family=spec.task_families[0],
        context_id=spec.context,
        action_kind=(
            AuthoringActionKind.TOOL if spec.required_tools else AuthoringActionKind.SKILL
        ),
        tool_or_skill_name=(
            spec.required_tools[0] if spec.required_tools else source.manifest.skill_id
        ),
        argument_schema_id="retain-readiness-arguments@1",
        observation_status=FailureMode.SUCCESS,
        token_bucket=TokenBucket.LE_1K,
        horizon_bucket=HorizonBucket.LE_3,
        absolute_log_importance=1.0,
        log_importance_quantile=1.0,
        invoked_skill_ids=(source.manifest.skill_id,),
        available_tools=spec.required_tools,
    )
    digest = stable_hash(
        {
            "base_seed": RETAIN_READINESS_BASE_SEED,
            "case_id": spec.name,
            "ordinal": ordinal,
            "suite_version": suite_version,
        }
    ).removeprefix("sha256:")
    return RetainReadinessCase(
        case_id=spec.name,
        profiles=spec.profiles,
        source=source,
        edge_exemplar=edge,
        evidence_summary=(
            "public synthetic high-flow, high-confidence Retain branch reliability evidence"
        ),
        seed=int(digest[:16], 16),
    )


def _repetitive(*sentences: str) -> str:
    core = " ".join(sentences)
    return "\n\n".join((core, core, core))


def _development_specs() -> tuple[_CaseSpec, ...]:
    return (
        _CaseSpec(
            name="dev-short-few-repetitive",
            profiles=(
                "few-requirements",
                "highly-repetitive",
                "short-instructions",
                "single-task-family",
                "without-tool",
            ),
            task_families=("synthetic/classification",),
            context="synthetic:short-repetitive",
            required_tools=(),
            title="Public classification discipline repeated for clarity",
            summary="Classify only from public fields and repeat no private inference.",
            instructions=_repetitive(
                "Read the public label set.",
                "Select one supported label.",
                "Do not infer hidden evaluation state.",
            ),
            requirement_texts=("Use only public input fields when selecting the label.",),
        ),
        _CaseSpec(
            name="dev-medium-many-tool",
            profiles=(
                "highly-repetitive",
                "many-requirements",
                "medium-instructions",
                "multiple-task-families",
                "with-tool",
            ),
            task_families=("synthetic/extract", "synthetic/lookup"),
            context="synthetic:medium-tool",
            required_tools=("debug.lookup",),
            title="Repeated public lookup and extraction procedure",
            summary="A verbose public lookup routine with repeated schema and stopping rules.",
            instructions=_repetitive(
                "Validate the public lookup key against the declared schema.",
                "Call debug.lookup once and preserve its public status.",
                "Extract only returned fields and stop after the public goal is met.",
                "Never retry a valid response or consult evaluator-only material.",
            ),
            requirement_texts=(
                "Validate the declared lookup key before the call.",
                "Call only debug.lookup with canonical JSON arguments.",
                "Use only fields in the public observation.",
                "Stop calling tools once the requested public field is available.",
            ),
        ),
        _CaseSpec(
            name="dev-long-few-structured-tool",
            profiles=(
                "few-requirements",
                "long-instructions",
                "single-task-family",
                "structured-low-redundancy",
                "with-tool",
            ),
            task_families=("synthetic/workflow",),
            context="synthetic:long-structured-tool",
            required_tools=("debug.lookup",),
            title="Ordered public workflow for one lookup and accountable completion",
            summary=(
                "An explicit low-redundancy workflow from public validation through completion."
            ),
            instructions="\n".join(
                (
                    "1. Identify the requested public field and its declared type.",
                    "2. Check that the lookup key is present without inventing a default.",
                    "3. Serialize exactly one debug.lookup request as canonical JSON.",
                    "4. Preserve parse and schema failures as observable agent behavior.",
                    "5. Distinguish an environment exception from a public tool error.",
                    "6. Read only public response fields and ignore evaluator-only state.",
                    "7. Compare returned evidence with the public goal before another action.",
                    "8. Avoid duplicate calls by carrying the completed request forward.",
                    "9. Complete concisely when the requested field is available.",
                    "10. Report a public failure without fabricating a successful value.",
                )
            ),
            requirement_texts=(
                "Perform exactly one schema-valid lookup and report only its public result.",
            ),
        ),
        _CaseSpec(
            name="dev-long-many-no-tool",
            profiles=(
                "highly-repetitive",
                "long-instructions",
                "many-requirements",
                "multiple-task-families",
                "without-tool",
            ),
            task_families=("synthetic/compare", "synthetic/summarize"),
            context="synthetic:long-no-tool",
            required_tools=(),
            title="Verbose public comparison and summary protocol",
            summary="A repeatedly stated protocol for comparing public records without tools.",
            instructions=_repetitive(
                "Normalize headings while preserving public values.",
                "Compare like fields in the supplied order.",
                "Mark missing public fields instead of filling them.",
                "Summarize agreements and differences concisely.",
                "Do not use answers, rewards, or private verifier data.",
            ),
            requirement_texts=(
                "Preserve the supplied record order.",
                "Compare only fields present in both public records.",
                "Mark missing values explicitly.",
                "Keep private evaluation material out of the summary.",
            ),
        ),
        _CaseSpec(
            name="dev-medium-few-structured-multi-tool",
            profiles=(
                "few-requirements",
                "medium-instructions",
                "multiple-task-families",
                "structured-low-redundancy",
                "with-tool",
            ),
            task_families=("synthetic/check", "synthetic/route"),
            context="synthetic:medium-structured-tool",
            required_tools=("debug.lookup",),
            title="Schema-first routing of one public lookup",
            summary="Validate, route, execute, observe, and finish a public lookup.",
            instructions=(
                "Validate the requested route and key. Construct one debug.lookup action with "
                "only declared arguments. Preserve the public observation status. Route a "
                "successful field to completion and report a public error as observed. Do not "
                "retry, replace the resource, or inspect private evaluator state."
            ),
            requirement_texts=("Use one declared lookup and preserve its public status.",),
        ),
        _CaseSpec(
            name="dev-short-many-structured",
            profiles=(
                "many-requirements",
                "short-instructions",
                "single-task-family",
                "structured-low-redundancy",
                "without-tool",
            ),
            task_families=("synthetic/format",),
            context="synthetic:short-many",
            required_tools=(),
            title="Public record formatter",
            summary="Format supplied public values without inference.",
            instructions="Validate fields, preserve order, format once, and report omissions.",
            requirement_texts=(
                "Validate public field names.",
                "Preserve supplied value order.",
                "Do not invent missing values.",
                "Return one concise formatted record.",
            ),
        ),
        _CaseSpec(
            name="dev-near-minimal",
            profiles=(
                "few-requirements",
                "near-minimal-legal",
                "short-instructions",
                "single-task-family",
                "structured-low-redundancy",
                "without-tool",
            ),
            task_families=("synthetic/minimal",),
            context="synthetic:near-minimal",
            required_tools=(),
            title="Public check",
            summary="Check public input.",
            instructions="Validate input; report its public status.",
            requirement_texts=("Use public input only.",),
        ),
    )


def _holdout_specs() -> tuple[_CaseSpec, ...]:
    return (
        _CaseSpec(
            name="holdout-short-few-repetitive-tool",
            profiles=(
                "few-requirements",
                "highly-repetitive",
                "short-instructions",
                "single-task-family",
                "with-tool",
            ),
            task_families=("synthetic/resolve",),
            context="synthetic:holdout-short-tool",
            required_tools=("debug.lookup",),
            title="Repeated public resolver rule",
            summary="Resolve one public key and repeat no hidden inference.",
            instructions=_repetitive(
                "Validate the public key.",
                "Call debug.lookup once.",
                "Report only the public response.",
            ),
            requirement_texts=("Resolve one declared public key without retry.",),
        ),
        _CaseSpec(
            name="holdout-medium-many-no-tool",
            profiles=(
                "highly-repetitive",
                "many-requirements",
                "medium-instructions",
                "multiple-task-families",
                "without-tool",
            ),
            task_families=("synthetic/group", "synthetic/order"),
            context="synthetic:holdout-medium-no-tool",
            required_tools=(),
            title="Repeated public grouping and ordering method",
            summary="Group public items while restating order and omission rules.",
            instructions=_repetitive(
                "Group items by the supplied public key.",
                "Keep first-seen order inside each group.",
                "Mark missing keys without inventing replacements.",
                "Return one concise public grouping.",
            ),
            requirement_texts=(
                "Use only the supplied grouping key.",
                "Preserve first-seen order.",
                "Mark missing keys.",
                "Return a concise grouping.",
            ),
        ),
        _CaseSpec(
            name="holdout-long-few-structured",
            profiles=(
                "few-requirements",
                "long-instructions",
                "multiple-task-families",
                "structured-low-redundancy",
                "without-tool",
            ),
            task_families=("synthetic/plan", "synthetic/review"),
            context="synthetic:holdout-long-structured",
            required_tools=(),
            title="Ordered public planning and review procedure",
            summary="A ten-stage public-only plan with explicit review and completion.",
            instructions="\n".join(
                (
                    "1. Read the public objective.",
                    "2. List supplied constraints.",
                    "3. Separate facts from assumptions.",
                    "4. Order feasible public steps.",
                    "5. Identify missing public inputs.",
                    "6. Avoid fabricating missing values.",
                    "7. Review each step against the objective.",
                    "8. Remove redundant steps.",
                    "9. State unresolved public gaps.",
                    "10. Return the concise reviewed plan.",
                )
            ),
            requirement_texts=("Produce a reviewed plan grounded only in public facts.",),
        ),
        _CaseSpec(
            name="holdout-long-many-tool",
            profiles=(
                "highly-repetitive",
                "long-instructions",
                "many-requirements",
                "single-task-family",
                "with-tool",
            ),
            task_families=("synthetic/audit",),
            context="synthetic:holdout-long-tool",
            required_tools=("debug.lookup",),
            title="Verbose public audit lookup procedure",
            summary="A redundant single-call audit routine for public data.",
            instructions=_repetitive(
                "Validate the public audit key and declared schema.",
                "Submit one canonical debug.lookup request.",
                "Preserve the returned public status and values.",
                "Do not retry or substitute another resource.",
                "Complete from observed public facts only.",
            ),
            requirement_texts=(
                "Validate the audit key.",
                "Use one canonical lookup.",
                "Preserve the public status.",
                "Complete without private material.",
            ),
        ),
        _CaseSpec(
            name="holdout-medium-few-structured-tool",
            profiles=(
                "few-requirements",
                "medium-instructions",
                "single-task-family",
                "structured-low-redundancy",
                "with-tool",
            ),
            task_families=("synthetic/inspect",),
            context="synthetic:holdout-medium-tool",
            required_tools=("debug.lookup",),
            title="Public inspection with one bounded lookup",
            summary="Inspect a declared public key and preserve the observation boundary.",
            instructions=(
                "Check the public key and argument type, call debug.lookup once, preserve parse "
                "or schema failures, use only returned public fields, avoid duplicate calls, "
                "and finish with a concise observed result."
            ),
            requirement_texts=("Inspect one public key with one declared lookup.",),
        ),
        _CaseSpec(
            name="holdout-short-many-structured-multi",
            profiles=(
                "many-requirements",
                "multiple-task-families",
                "short-instructions",
                "structured-low-redundancy",
                "without-tool",
            ),
            task_families=("synthetic/filter", "synthetic/select"),
            context="synthetic:holdout-short-many",
            required_tools=(),
            title="Public selector",
            summary="Filter and select from public values.",
            instructions="Validate, filter, preserve order, and select once.",
            requirement_texts=(
                "Validate public candidates.",
                "Apply only the supplied filter.",
                "Preserve candidate order.",
                "Return one supported selection.",
            ),
        ),
        _CaseSpec(
            name="holdout-near-minimal",
            profiles=(
                "few-requirements",
                "near-minimal-legal",
                "short-instructions",
                "single-task-family",
                "structured-low-redundancy",
                "without-tool",
            ),
            task_families=("synthetic/compact",),
            context="synthetic:holdout-near-minimal",
            required_tools=(),
            title="Public verify",
            summary="Verify supplied data.",
            instructions="Check public data; state the observed result.",
            requirement_texts=("Use supplied public data.",),
        ),
    )


def retain_readiness_suite(kind: RetainSuiteKind) -> RetainReadinessSuite:
    if kind is RetainSuiteKind.DEVELOPMENT:
        specs = _development_specs()
        version = RETAIN_DEVELOPMENT_SUITE_VERSION
    elif kind is RetainSuiteKind.HOLDOUT:
        specs = _holdout_specs()
        version = RETAIN_HOLDOUT_SUITE_VERSION
    else:
        raise TypeError("unsupported Retain readiness suite")
    return RetainReadinessSuite(
        kind=kind,
        version=version,
        cases=tuple(
            _case(spec, suite_version=version, ordinal=index)
            for index, spec in enumerate(specs, start=1)
        ),
    )


__all__ = [
    "RETAIN_AUTHORING_SAMPLING",
    "RETAIN_DEVELOPMENT_SUITE_VERSION",
    "RETAIN_HOLDOUT_SUITE_VERSION",
    "RETAIN_READINESS_SUITE_FORMAT",
    "RetainReadinessCase",
    "RetainReadinessSuite",
    "RetainSuiteKind",
    "retain_readiness_suite",
]
