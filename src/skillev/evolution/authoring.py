"""Sealed, one-shot frozen-base skill authoring from public evidence."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Protocol, TypeAlias

from skillev.contracts import (
    GenerateEvidence,
    JsonValue,
    SplitModalityEvidence,
    canonical_json,
    stable_hash,
)
from skillev.policy import (
    AuthoringGenerationRequest,
    AuthoringTokenizerProtocol,
    PolicyBackbone,
)
from skillev.runtime import (
    BudgetLedger,
    BudgetReservation,
    BudgetSettlement,
    BudgetVector,
    EventType,
    RuntimeEventEmitter,
    SkillApplicability,
    SkillDocument,
    SkillRequirement,
)

from .authoring_validation import AuthoringFailedError, validate_authoring_result
from .config import EvolutionConfig
from .evidence import AuthoringEdgeEvidence
from .requirement_contract import immutable_requirements, revision_contract

AUTHORING_TEMPLATE_RETAIN = "skill-retain-compress@9"
AUTHORING_TEMPLATE_REFINE = "skill-refine@7"
AUTHORING_TEMPLATE_SPLIT = "skill-split@8"
AUTHORING_TEMPLATE_GENERATE = "skill-generate@9"


class RetainCompressionInfeasibleError(AuthoringFailedError):
    """The source is already at the exact minimum legal retained shape."""


@dataclass(frozen=True, slots=True)
class RetainCompressionBudget:
    """Exact tokenizer-relative feasibility and output bound for Retain."""

    source_model_visible_token_count: int
    minimum_legal_model_visible_token_count: int
    maximum_output_model_visible_token_count: int

    def __post_init__(self) -> None:
        values = (
            self.source_model_visible_token_count,
            self.minimum_legal_model_visible_token_count,
            self.maximum_output_model_visible_token_count,
        )
        if any(type(value) is not int or value < 1 for value in values):
            raise ValueError("Retain compression token counts must be positive integers")
        if self.maximum_output_model_visible_token_count != (
            self.source_model_visible_token_count - 1
        ):
            raise ValueError("Retain output maximum must be exactly source minus one")
        if self.minimum_legal_model_visible_token_count >= (self.source_model_visible_token_count):
            raise RetainCompressionInfeasibleError(
                "Retain source has no strictly smaller legal model-visible representation"
            )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "maximum_output_model_visible_token_count": (
                self.maximum_output_model_visible_token_count
            ),
            "minimum_legal_model_visible_token_count": (
                self.minimum_legal_model_visible_token_count
            ),
            "source_model_visible_token_count": self.source_model_visible_token_count,
        }


@dataclass(frozen=True, slots=True)
class AuthoringSamplingConfig:
    temperature: float
    top_p: float

    def __post_init__(self) -> None:
        for field in ("temperature", "top_p"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{field} must be finite")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{field} must be finite")
            object.__setattr__(self, field, number)
        if self.temperature <= 0.0 or not 0.0 < self.top_p <= 1.0:
            raise ValueError("authoring sampling controls are out of range")

    def to_value(self) -> dict[str, JsonValue]:
        return {"temperature": self.temperature, "top_p": self.top_p}

    @classmethod
    def from_value(cls, value: object) -> AuthoringSamplingConfig:
        if not isinstance(value, dict) or set(value) != {"temperature", "top_p"}:
            raise ValueError("AuthoringSamplingConfig has incompatible fields")
        temperature = value["temperature"]
        top_p = value["top_p"]
        if any(
            isinstance(item, bool) or not isinstance(item, int | float)
            for item in (temperature, top_p)
        ):
            raise TypeError("authoring sampling controls must be numeric")
        return cls(temperature=float(temperature), top_p=float(top_p))


@dataclass(frozen=True, slots=True)
class AuthoringCallMaximum:
    input_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        if type(self.input_tokens) is not int or self.input_tokens < 1:
            raise ValueError("authoring input maximum must be positive")
        if type(self.output_tokens) is not int or self.output_tokens < 1:
            raise ValueError("authoring output maximum must be positive")

    def to_budget_vector(self) -> BudgetVector:
        return BudgetVector(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            model_calls=1,
        )


@dataclass(frozen=True, slots=True)
class GenerateAuthoritySelection:
    task_family: str
    context: str
    required_tools: tuple[str, ...]
    input_schema_id: str
    output_schema_id: str
    license_id: str

    def __post_init__(self) -> None:
        for field in (
            "task_family",
            "context",
            "input_schema_id",
            "output_schema_id",
            "license_id",
        ):
            if not getattr(self, field).strip():
                raise ValueError(f"{field} must be non-empty text")
        _sorted_unique_text(self.required_tools, field="required_tools")


@dataclass(frozen=True, slots=True)
class SkillAuthoringAuthority:
    """Closed authoring authority and complete finite Split family universe."""

    input_schema_id: str
    output_schema_id: str
    license_id: str
    allowed_task_families: tuple[str, ...]
    allowed_tools: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in ("input_schema_id", "output_schema_id", "license_id"):
            if not getattr(self, field).strip():
                raise ValueError(f"{field} must be non-empty text")
        _sorted_unique_text(
            self.allowed_task_families,
            field="allowed_task_families",
        )
        _sorted_unique_text(self.allowed_tools, field="allowed_tools")
        if not self.allowed_task_families:
            raise ValueError("authoring authority requires task families")
        if "*" in self.allowed_task_families:
            raise ValueError("authoring authority task-family universe must be finite")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "allowed_task_families": list(self.allowed_task_families),
            "allowed_tools": list(self.allowed_tools),
            "input_schema_id": self.input_schema_id,
            "license_id": self.license_id,
            "output_schema_id": self.output_schema_id,
        }

    @classmethod
    def from_value(cls, value: object) -> SkillAuthoringAuthority:
        fields = {
            "allowed_task_families",
            "allowed_tools",
            "input_schema_id",
            "license_id",
            "output_schema_id",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("SkillAuthoringAuthority has incompatible fields")
        families = value["allowed_task_families"]
        tools = value["allowed_tools"]
        if not isinstance(families, list) or any(type(item) is not str for item in families):
            raise TypeError("allowed_task_families must be a text array")
        if not isinstance(tools, list) or any(type(item) is not str for item in tools):
            raise TypeError("allowed_tools must be a text array")
        for field in ("input_schema_id", "license_id", "output_schema_id"):
            if type(value[field]) is not str:
                raise TypeError(f"{field} must be text")
        return cls(
            input_schema_id=value["input_schema_id"],
            output_schema_id=value["output_schema_id"],
            license_id=value["license_id"],
            allowed_task_families=tuple(families),
            allowed_tools=tuple(tools),
        )

    def select(
        self,
        exemplars: tuple[AuthoringEdgeEvidence, ...],
    ) -> GenerateAuthoritySelection:
        if not exemplars:
            raise AuthoringFailedError("Generate requires uncovered edge exemplars")
        task_families = {item.task_family for item in exemplars}
        contexts = {item.context_id for item in exemplars}
        tool_sets = {item.available_tools for item in exemplars}
        if len(task_families) != 1 or len(contexts) != 1 or len(tool_sets) != 1:
            raise AuthoringFailedError("Generate evidence has no unique authority")
        task_family = next(iter(task_families))
        context = next(iter(contexts))
        tools = next(iter(tool_sets))
        if task_family not in self.allowed_task_families:
            raise AuthoringFailedError("Generate task family is outside authority")
        if not set(tools) <= set(self.allowed_tools):
            raise AuthoringFailedError("Generate tools are outside authority")
        return GenerateAuthoritySelection(
            task_family=task_family,
            context=context,
            required_tools=tools,
            input_schema_id=self.input_schema_id,
            output_schema_id=self.output_schema_id,
            license_id=self.license_id,
        )


@dataclass(frozen=True, slots=True)
class AuthoredSkillDraft:
    title: str
    summary: str
    instructions: str
    applicability: SkillApplicability
    requirements: tuple[SkillRequirement, ...]

    def __post_init__(self) -> None:
        if not self.title.strip() or not self.summary.strip() or not self.instructions.strip():
            raise ValueError("authored text fields cannot be empty")
        if not isinstance(self.applicability, SkillApplicability):
            raise TypeError("draft applicability must be SkillApplicability")
        if not self.requirements or any(
            not isinstance(item, SkillRequirement) for item in self.requirements
        ):
            raise ValueError("draft requirements must be non-empty")
        if len({item.requirement_id for item in self.requirements}) != len(self.requirements):
            raise ValueError("draft requirement IDs must be unique")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "applicability": self.applicability.to_value(),
            "instructions": self.instructions,
            "requirements": [item.to_value() for item in self.requirements],
            "summary": self.summary,
            "title": self.title,
        }

    @classmethod
    def from_value(cls, value: object) -> AuthoredSkillDraft:
        if not isinstance(value, dict) or set(value) != {
            "applicability",
            "instructions",
            "requirements",
            "summary",
            "title",
        }:
            raise ValueError("AuthoredSkillDraft has incompatible fields")
        for field in ("instructions", "summary", "title"):
            if not isinstance(value[field], str):
                raise ValueError(f"draft {field} must be text")
        requirements = value["requirements"]
        if not isinstance(requirements, list):
            raise ValueError("draft requirements must be an array")
        return cls(
            title=value["title"],
            summary=value["summary"],
            instructions=value["instructions"],
            applicability=SkillApplicability.from_value(value["applicability"]),
            requirements=tuple(SkillRequirement.from_value(item) for item in requirements),
        )

    @classmethod
    def from_document(cls, document: SkillDocument) -> AuthoredSkillDraft:
        if not isinstance(document, SkillDocument):
            raise TypeError("document must be SkillDocument")
        return cls(
            title=document.title,
            summary=document.summary,
            instructions=document.instructions,
            applicability=document.applicability,
            requirements=document.requirements,
        )


@dataclass(frozen=True, slots=True)
class RetainAuthoringRequest:
    source: SkillDocument
    edge_exemplars: tuple[AuthoringEdgeEvidence, ...]
    evidence_summary: str
    seed: int

    def __post_init__(self) -> None:
        _request_common(self.edge_exemplars, self.evidence_summary, self.seed)


@dataclass(frozen=True, slots=True)
class RefineAuthoringRequest:
    source: SkillDocument
    edge_exemplars: tuple[AuthoringEdgeEvidence, ...]
    evidence_summary: str
    target_context_keys: tuple[str, ...]
    seed: int

    def __post_init__(self) -> None:
        _request_common(self.edge_exemplars, self.evidence_summary, self.seed)
        _sorted_unique_text(self.target_context_keys, field="target_context_keys")
        if not self.target_context_keys:
            raise ValueError("Refine requires target context keys")


@dataclass(frozen=True, slots=True)
class SplitAuthoringRequest:
    source: SkillDocument
    edge_exemplars: tuple[AuthoringEdgeEvidence, ...]
    evidence_summary: str
    modality: SplitModalityEvidence
    seed: int

    def __post_init__(self) -> None:
        _request_common(self.edge_exemplars, self.evidence_summary, self.seed)
        if not isinstance(self.modality, SplitModalityEvidence):
            raise ValueError("Split requires modality evidence")


@dataclass(frozen=True, slots=True)
class GenerateAuthoringRequest:
    edge_exemplars: tuple[AuthoringEdgeEvidence, ...]
    evidence_summary: str
    authority: GenerateAuthoritySelection
    evidence: GenerateEvidence
    seed: int
    related_skills: tuple[SkillDocument, ...] = ()

    def __post_init__(self) -> None:
        _request_common(self.edge_exemplars, self.evidence_summary, self.seed)
        if not self.edge_exemplars:
            raise ValueError("Generate requires edge exemplars")
        if not isinstance(self.authority, GenerateAuthoritySelection):
            raise ValueError("Generate requires an authority selection")
        if not isinstance(self.evidence, GenerateEvidence):
            raise ValueError("Generate requires sealed selection evidence")
        if tuple(item.edge_id for item in self.edge_exemplars) != self.evidence.importance_edge_ids:
            raise ValueError("Generate request exemplars differ from selection evidence")


AuthoringRequest: TypeAlias = (
    RetainAuthoringRequest
    | RefineAuthoringRequest
    | SplitAuthoringRequest
    | GenerateAuthoringRequest
)


@dataclass(frozen=True, slots=True)
class AuthoringResult:
    drafts: tuple[AuthoredSkillDraft, ...]

    def __post_init__(self) -> None:
        if not self.drafts or any(not isinstance(item, AuthoredSkillDraft) for item in self.drafts):
            raise ValueError("authoring result requires structured drafts")

    def to_value(self) -> dict[str, JsonValue]:
        return {"drafts": [item.to_value() for item in self.drafts]}

    @classmethod
    def from_value(cls, value: object) -> AuthoringResult:
        if not isinstance(value, dict) or set(value) != {"drafts"}:
            raise ValueError("AuthoringResult has incompatible fields")
        drafts = value["drafts"]
        if not isinstance(drafts, list):
            raise ValueError("AuthoringResult.drafts must be an array")
        return cls(drafts=tuple(AuthoredSkillDraft.from_value(item) for item in drafts))


class SkillAuthor(Protocol):
    @property
    def tokenizer(self) -> AuthoringTokenizerProtocol: ...

    def author(self, request: AuthoringRequest) -> AuthoringResult: ...


class BaseModelSkillAuthor:
    def __init__(
        self,
        *,
        backbone: PolicyBackbone,
        config: EvolutionConfig,
        sampling: AuthoringSamplingConfig,
        ledger: BudgetLedger,
        emitter: RuntimeEventEmitter,
        maximum: AuthoringCallMaximum,
    ) -> None:
        self._backbone = backbone
        self._config = config
        self._sampling = sampling
        self._ledger = ledger
        self._emitter = emitter
        self._maximum = maximum
        if maximum.input_tokens != config.max_authoring_prompt_tokens:
            raise ValueError("authoring input maximum differs from EvolutionConfig")
        if maximum.output_tokens != config.max_authoring_completion_tokens:
            raise ValueError("authoring output maximum differs from EvolutionConfig")

    @property
    def tokenizer(self) -> AuthoringTokenizerProtocol:
        return self._backbone.tokenizer

    def author(self, request: AuthoringRequest) -> AuthoringResult:
        from .authoring_material import render_bounded_authoring_prompt

        prompt = render_bounded_authoring_prompt(
            request, tokenizer=self.tokenizer, maximum_tokens=self._maximum.input_tokens
        )
        input_ids = tuple(self.tokenizer.encode_authoring_prompt(prompt))
        if not input_ids:
            raise AuthoringFailedError("authoring prompt encoded to no tokens")
        if len(input_ids) > self._maximum.input_tokens:
            raise AuthoringFailedError("authoring prompt exceeds its fixed maximum")
        template_version = template_version_for(request)
        output_token_maximum = self._maximum.output_tokens
        if isinstance(request, RetainAuthoringRequest):
            output_token_maximum = min(
                output_token_maximum,
                retain_generation_token_limit(request.source, tokenizer=self.tokenizer),
            )
        reservation = BudgetReservation(
            reservation_id=authoring_reservation_id(request),
            run_id=self._ledger.run_id,
            attempt_id=self._ledger.attempt_id,
            invocation_id=f"phi-authoring:{request.seed}",
            maximum=BudgetVector(
                input_tokens=len(input_ids),
                output_tokens=output_token_maximum,
                model_calls=1,
            ),
        )
        self._ledger.reserve(reservation)
        self._emitter.emit(
            EventType.BUDGET_RESERVED,
            {
                "maximum": reservation.maximum.to_value(),
                "reservation_id": reservation.reservation_id,
            },
        )
        result = self._backbone.generate_base(
            AuthoringGenerationRequest(
                input_ids=input_ids,
                max_new_tokens=output_token_maximum,
                temperature=self._sampling.temperature,
                top_p=self._sampling.top_p,
                seed=request.seed,
                template_version=template_version,
                completion_boundary_version=(self._config.authoring_completion_boundary_version),
            )
        )
        actual_usage = BudgetVector(
            input_tokens=len(input_ids),
            output_tokens=len(result.content_token_ids) + len(result.stop_token_ids),
            model_calls=1,
        )
        self._ledger.settle(
            BudgetSettlement(
                reservation_id=reservation.reservation_id,
                actual=actual_usage,
            )
        )
        self._emitter.emit(
            EventType.BUDGET_SETTLED,
            {
                "actual": actual_usage.to_value(),
                "reservation_id": reservation.reservation_id,
            },
        )
        if not result.content_token_ids:
            raise AuthoringFailedError("frozen base returned no content")
        try:
            raw = json.loads(self.tokenizer.decode(result.content_token_ids))
            authored = AuthoringResult.from_value(raw)
            validate_authoring_result(
                request,
                authored,
                tokenizer=self.tokenizer,
                max_skill_instruction_tokens_per_draft=(
                    self._config.max_skill_instruction_tokens_per_draft
                ),
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise AuthoringFailedError("authoring output violates its sealed schema") from error
        return authored


def template_version_for(request: AuthoringRequest) -> str:
    match request:
        case RetainAuthoringRequest():
            return AUTHORING_TEMPLATE_RETAIN
        case RefineAuthoringRequest():
            return AUTHORING_TEMPLATE_REFINE
        case SplitAuthoringRequest():
            return AUTHORING_TEMPLATE_SPLIT
        case GenerateAuthoringRequest():
            return AUTHORING_TEMPLATE_GENERATE
        case _:
            raise TypeError(f"unsupported authoring request: {type(request).__name__}")


def authoring_reservation_id(request: AuthoringRequest) -> str:
    """Return the deterministic ledger identity for one sealed authoring call."""

    return f"phi-authoring:{template_version_for(request)}:{request.seed}"


def minimum_legal_retain_draft(source: SkillDocument) -> AuthoredSkillDraft:
    """Return the fixed canonical lower-bound shape for one retained skill.

    Retain must preserve complete applicability and every requirement verbatim.
    The only author-controlled fields may shrink to one non-whitespace character;
    this fixed shape therefore gives the pre-generation method rule rather than a
    post-generation fallback.
    """

    if not isinstance(source, SkillDocument):
        raise TypeError("Retain source must be SkillDocument")
    return AuthoredSkillDraft(
        title="x",
        summary="x",
        instructions="x",
        applicability=source.applicability,
        requirements=source.requirements,
    )


def retain_compression_budget(
    source: SkillDocument,
    *,
    tokenizer: AuthoringTokenizerProtocol,
) -> RetainCompressionBudget:
    """Compute and enforce the exact pre-generation Retain feasibility rule."""

    source_tokens = len(tokenizer.encode(canonical_json(source.content_value())))
    minimum_tokens = len(
        tokenizer.encode(canonical_json(minimum_legal_retain_draft(source).to_value()))
    )
    return RetainCompressionBudget(
        source_model_visible_token_count=source_tokens,
        minimum_legal_model_visible_token_count=minimum_tokens,
        maximum_output_model_visible_token_count=source_tokens - 1,
    )


def retain_generation_token_limit(
    source: SkillDocument,
    *,
    tokenizer: AuthoringTokenizerProtocol,
) -> int:
    """Bound raw JSON generation tightly enough to make Retain executable.

    The scientific compression rule counts the canonical draft.  Generation
    additionally needs the fixed ``{"drafts":[...]}`` envelope, so this bound
    adds exactly that envelope's tokenizer cost around the canonical minimum.
    The sealed validator remains authoritative after the one model call.
    """

    budget = retain_compression_budget(source, tokenizer=tokenizer)
    minimum = minimum_legal_retain_draft(source)
    minimum_draft_tokens = len(tokenizer.encode(canonical_json(minimum.to_value())))
    minimum_result_tokens = len(
        tokenizer.encode(canonical_json(AuthoringResult((minimum,)).to_value()))
    )
    envelope_tokens = max(1, minimum_result_tokens - minimum_draft_tokens)
    return budget.maximum_output_model_visible_token_count + envelope_tokens


def render_retain_prompt(
    request: RetainAuthoringRequest,
    *,
    tokenizer: AuthoringTokenizerProtocol,
) -> str:
    budget = retain_compression_budget(request.source, tokenizer=tokenizer)
    contract = _common_output_contract(draft_count=1)
    contract["action_constraints"] = {
        "exact_applicability": request.source.applicability.to_value(),
        "exact_requirement_ids": [item.requirement_id for item in request.source.requirements],
        "exact_requirements": [item.to_value() for item in request.source.requirements],
        "compression_measure": "complete canonical model-visible skill content",
        "exact_authored_text_types": {
            "instructions": "JSON string",
            "requirement.text": "JSON string",
            "summary": "JSON string",
            "title": "JSON string",
        },
        **budget.to_value(),
        "maximum_generation_content_token_count": retain_generation_token_limit(
            request.source,
            tokenizer=tokenizer,
        ),
        "tokenizer_identity": tokenizer.public_identity.to_value(),
    }
    return _render_prompt(
        template=AUTHORING_TEMPLATE_RETAIN,
        material={
            "edge_exemplars": [item.to_value() for item in request.edge_exemplars],
            "evidence_summary": request.evidence_summary,
            "source": request.source.content_value(),
        },
        contract=contract,
        directive=(
            "Return one concise complete skill. Copy exact_applicability and "
            "exact_requirements verbatim. Remove repetition from title, summary and "
            "instructions without discarding executable constraints. The tokenizer "
            "named in the contract "
            "must count the complete canonical draft at or below "
            "maximum_output_model_visible_token_count. This is a hard validity bound "
            "on title, summary, instructions, requirement texts, and JSON structure "
            "together—not only on instructions. Do not repeat evidence or add commentary. "
            "Use concise imperative prose, preserving the source capability and all "
            "execution constraints; requirement texts are immutable. "
            "Return minified JSON with no optional whitespace and finish within "
            "maximum_generation_content_token_count. The root must have only drafts; "
            "each draft must have exactly applicability, instructions, requirements, "
            "summary, and title; copy each exact requirement including its kind and "
            "revision metadata when present. Title, summary, instructions, and every "
            "requirement text must each be "
            "one JSON string—never an array, object, or list of steps. Join multiple "
            "imperatives into one compact string. Never put token counts, compliance "
            "notes, metadata, or explanations inside the output object."
        ),
    )


def render_refine_prompt(request: RefineAuthoringRequest) -> str:
    contract = _common_output_contract(draft_count=1)
    contract["action_constraints"] = {
        "applicability_by_draft": [request.source.applicability.to_value()],
        **revision_contract(request.source),
        "required_requirement_ids_by_draft": [
            [
                *[item.requirement_id for item in immutable_requirements(request.source)],
                *[context_patch_requirement_id(key) for key in request.target_context_keys],
            ]
        ],
    }
    return _render_prompt(
        template=AUTHORING_TEMPLATE_REFINE,
        material={
            "edge_exemplars": [item.to_value() for item in request.edge_exemplars],
            "evidence_summary": request.evidence_summary,
            "source": request.source.content_value(),
            "target_context_keys": list(request.target_context_keys),
        },
        contract=contract,
        directive=(
            "Return one refined skill preserving immutable constraints. Correct strategy "
            "errors using the weak-context evidence; include every context-patch strategy. "
            "Follow strategy_revision_rule for all changed or removed strategies."
        ),
    )


def render_split_prompt(request: SplitAuthoringRequest) -> str:
    source_ids = [item.requirement_id for item in immutable_requirements(request.source)]
    low_applicability, high_applicability = split_applicabilities(request)
    low_families = request.modality.low_task_families
    high_families = request.modality.high_task_families
    contract = _common_output_contract(draft_count=2)
    contract["action_constraints"] = {
        # Exact complete applicability objects leave no authoring freedom to alter
        # the independent retrieval-context, exclusion, or tool axes.
        "applicability_by_draft": [
            low_applicability.to_value(),
            high_applicability.to_value(),
        ],
        **revision_contract(request.source),
        "partition_axis": "task_families",
        "posterior_context_semantics": "ContextFeature.context == task_family",
        "required_requirement_ids_by_draft": [
            [
                *source_ids,
                split_task_family_requirement_id(low_families),
            ],
            [
                *source_ids,
                split_task_family_requirement_id(high_families),
            ],
        ],
    }
    return _render_prompt(
        template=AUTHORING_TEMPLATE_SPLIT,
        material={
            "edge_exemplars": [item.to_value() for item in request.edge_exemplars],
            "evidence_summary": request.evidence_summary,
            "exact_task_family_partition": {
                "high": list(high_families),
                "low": list(low_families),
            },
            "modality": request.modality.to_value(),
            "source": request.source.content_value(),
        },
        contract=contract,
        directive=(
            "Return exactly two drafts in low-mode then high-mode order. Partition "
            "only task_families according to the posterior modes; preserve contexts, "
            "excluded_contexts, and required_tools exactly. Preserve immutable constraints, "
            "but specialize strategies to the child's supported modality; explain every "
            "removed or corrected strategy using strategy_revision_rule."
        ),
    )


def generate_output_constraints(request: GenerateAuthoringRequest) -> dict[str, JsonValue]:
    """One authoritative structural projection for Generate prompt and wire schema."""
    return {
        "applicability_by_draft": [
            {
                "contexts": [request.authority.context],
                "excluded_contexts": [],
                "required_tools": list(request.authority.required_tools),
                "task_families": [request.authority.task_family],
            }
        ],
        "required_requirement_ids_by_draft": [
            [uncovered_edge_requirement_id(item.edge_id) for item in request.edge_exemplars]
        ],
    }


def render_generate_prompt(request: GenerateAuthoringRequest) -> str:
    contract = _common_output_contract(draft_count=1)
    contract["requirement_fields"] = ["requirement_id", "text", "kind"]
    contract["optional_requirement_fields"] = ["replaces"]
    contract["requirement_kind_rule"] = (
        "Every Generate requirement must explicitly set kind=evolvable-strategy. "
        "Generate creates new strategies, never immutable authority or revisions; "
        "omit replaces or leave it empty, and do not emit change_reason. "
        "Return exactly one draft, copy every applicability field exactly (including "
        "empty required_tools), and emit exactly the required requirement IDs in order."
    )
    contract["action_constraints"] = generate_output_constraints(request)
    return _render_prompt(
        template=AUTHORING_TEMPLATE_GENERATE,
        material={
            "related_skills": [
                {
                    "skill_id": document.manifest.skill_id,
                    "summary": document.summary,
                    "applicability": document.applicability.to_value(),
                }
                for document in request.related_skills
            ],
            "coverage_semantics": "not-explicitly-invoked; exposure-is-not-credit",
            "edge_exemplars": [item.to_value() for item in request.edge_exemplars],
            "evidence_summary": request.evidence_summary,
            "selection_rule": {
                "importance_quantile": request.evidence.importance_quantile,
                "importance_semantics": request.evidence.importance_semantics,
                "minimum_absolute_log_importance": (
                    request.evidence.minimum_absolute_log_importance
                ),
            },
        },
        contract=contract,
        directive=(
            "Return one new skill with the exact authoritative applicability and an "
            "explicit requirement for every uncovered edge. Compare related existing skills "
            "and add a reusable missing capability, not a task answer. An unused "
            "exposed skill is not proof that no applicable skill existed."
        ),
    )


def render_authoring_prompt(
    request: AuthoringRequest,
    *,
    tokenizer: AuthoringTokenizerProtocol,
) -> str:
    match request:
        case RetainAuthoringRequest():
            return render_retain_prompt(request, tokenizer=tokenizer)
        case RefineAuthoringRequest():
            return render_refine_prompt(request)
        case SplitAuthoringRequest():
            return render_split_prompt(request)
        case GenerateAuthoringRequest():
            return render_generate_prompt(request)
        case _:
            raise TypeError(f"unsupported authoring request: {type(request).__name__}")


def _render_prompt(
    *,
    template: str,
    material: dict[str, JsonValue],
    contract: dict[str, JsonValue],
    directive: str,
) -> str:
    payload: dict[str, JsonValue] = {
        "material": material,
        "output_contract": contract,
        "template_version": template,
    }
    return (
        "Use only this public model-visible authoring material:\n"
        f"{canonical_json(payload)}\n"
        f"{directive} "
        "Return exactly one JSON object with root field drafts. Do not emit markdown, "
        "answers, reward, verifier data, native payload, or explanations.\n"
    )


def _common_output_contract(*, draft_count: int) -> dict[str, JsonValue]:
    return {
        "text_rule": (
            "Title, summary, instructions and requirement text must contain non-whitespace "
            "content. Encode quotes, backslashes and newlines with ordinary JSON escapes. "
            "A strategy revision's change_reason must also contain non-whitespace content."
        ),
        "applicability_fields": [
            "contexts",
            "excluded_contexts",
            "required_tools",
            "task_families",
        ],
        "draft_count": draft_count,
        "draft_fields": [
            "applicability",
            "instructions",
            "requirements",
            "summary",
            "title",
        ],
        "requirement_fields": ["requirement_id", "text"],
        "optional_requirement_fields": ["kind", "replaces", "change_reason"],
        "requirement_kind_rule": (
            "Every new strategy/patch must set kind=evolvable-strategy. Missing kind "
            "means historical immutable-constraint; copy source constraints exactly. "
            "replaces and change_reason are used only for evidence-based strategy revisions."
        ),
    }


def context_patch_requirement_id(context_key: str) -> str:
    return _derived_requirement_id("context-patch", context_key)


def split_task_family_requirement_id(task_families: tuple[str, ...]) -> str:
    return _derived_requirement_id("split-task-family", "\n".join(task_families))


def split_applicabilities(
    request: SplitAuthoringRequest,
) -> tuple[SkillApplicability, SkillApplicability]:
    """Seal the two Split drafts to a task-family-only partition.

    The source document's retrieval context axis is intentionally copied exactly;
    posterior ``ContextFeature.context`` is task-family conditioned under the
    fixed protocol and is never a retrieval ``context_id``.
    """

    source = request.source.applicability
    if source.task_families != ("*",) and source.task_families != (
        request.modality.source_task_families
    ):
        raise AuthoringFailedError("Split evidence differs from source applicability")

    def for_families(task_families: tuple[str, ...]) -> SkillApplicability:
        return SkillApplicability(
            task_families=task_families,
            contexts=source.contexts,
            required_tools=source.required_tools,
            excluded_contexts=source.excluded_contexts,
        )

    return for_families(request.modality.low_task_families), for_families(
        request.modality.high_task_families
    )


def uncovered_edge_requirement_id(edge_id: str) -> str:
    return _derived_requirement_id("uncovered-edge", edge_id)


def _derived_requirement_id(prefix: str, value: str) -> str:
    digest = stable_hash({"kind": prefix, "value": value}).removeprefix("sha256:")
    return f"{prefix}-{digest[:12]}"


def _request_common(
    edge_exemplars: tuple[AuthoringEdgeEvidence, ...],
    evidence_summary: str,
    seed: int,
) -> None:
    if any(not isinstance(item, AuthoringEdgeEvidence) for item in edge_exemplars):
        raise ValueError("edge_exemplars must contain AuthoringEdgeEvidence values")
    ids = tuple(item.edge_id for item in edge_exemplars)
    if len(set(ids)) != len(ids):
        raise ValueError("edge_exemplars must have unique edge IDs")
    if not evidence_summary.strip():
        raise ValueError("evidence_summary cannot be empty")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")


def _sorted_unique_text(values: tuple[str, ...], *, field: str) -> None:
    if any(not isinstance(item, str) or not item.strip() for item in values):
        raise ValueError(f"{field} must contain non-empty text")
    if tuple(sorted(set(values))) != values:
        raise ValueError(f"{field} must be sorted and unique")


__all__ = [
    "AUTHORING_TEMPLATE_GENERATE",
    "AUTHORING_TEMPLATE_REFINE",
    "AUTHORING_TEMPLATE_RETAIN",
    "AUTHORING_TEMPLATE_SPLIT",
    "AuthoredSkillDraft",
    "AuthoringCallMaximum",
    "AuthoringFailedError",
    "AuthoringRequest",
    "AuthoringResult",
    "AuthoringSamplingConfig",
    "BaseModelSkillAuthor",
    "GenerateAuthoringRequest",
    "GenerateAuthoritySelection",
    "RefineAuthoringRequest",
    "RetainAuthoringRequest",
    "RetainCompressionBudget",
    "RetainCompressionInfeasibleError",
    "SkillAuthor",
    "SkillAuthoringAuthority",
    "SplitAuthoringRequest",
    "context_patch_requirement_id",
    "minimum_legal_retain_draft",
    "render_authoring_prompt",
    "render_generate_prompt",
    "render_refine_prompt",
    "render_retain_prompt",
    "render_split_prompt",
    "retain_compression_budget",
    "retain_generation_token_limit",
    "split_applicabilities",
    "split_task_family_requirement_id",
    "template_version_for",
    "uncovered_edge_requirement_id",
    "validate_authoring_result",
]
