"""Deterministic assembly of the public initial rollout context ``H_0``."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final, Protocol

from skillev.contracts import (
    InitialContext,
    JsonValue,
    TokenizerProtocol,
    canonical_json,
    normalize_json,
)
from skillev.contracts.action_wire import (
    ACTION_WIRES,
    NATIVE_TOOL_CARRIER_WIRE,
    NATIVE_TOOL_HANDOFF_WIRE,
    NATIVE_TOOL_WIRES,
)
from skillev.contracts.skill_exposure import (
    CATALOG_EXPOSURES,
    PROACTIVE_CATALOG_EXPOSURE,
    SKILL_EXPOSURES,
    TWO_SKILL_CATALOG_EXPOSURE,
)
from skillev.policy.interface import (
    INPUT_WINDOW_META_KEY,
    OBSERVED_PUBLIC_HISTORY,
    ROLLOUT_SOURCE_MESSAGES_BEGIN,
    ROLLOUT_SOURCE_MESSAGES_END,
    TYPED_PUBLIC_HISTORY,
    ModelInputWindow,
    PhaseContextSpec,
    encode_rollout_prompt,
)
from skillev.runtime.full_skill_context import FullRetrievedSkillContext
from skillev.scoring import assembled_context_hash
from skillev.task_semantic_guidance import (
    LEGACY_TASK_SEMANTICS,
    PUBLIC_TASK_SEMANTICS_V2,
    PUBLIC_TASK_SEMANTICS_V3,
    PUBLIC_TASK_SEMANTICS_V4,
    PUBLIC_TASK_SEMANTICS_V5,
    PUBLIC_TASK_SEMANTICS_V7,
    PUBLIC_TASK_SEMANTICS_V8,
    PUBLIC_TASK_SEMANTICS_V9,
    PUBLIC_TASK_SEMANTICS_V10,
    TRAINING_PUBLIC_INPUT,
    TRAINING_SUBMISSION_INSTRUCTION,
    phase_deliverable,
    public_task_semantics,
    validate_task_semantic_guidance,
)

from .action_contract import (
    ActionContract,
)
from .action_contract import (
    action_example as _action_example,
)
from .action_contract import (
    completion_example as _completion_example,
)
from .action_surface import ACTION_SURFACE_FORMAT_V3, ActionSurface, PublicActionInstructions
from .generator import RolloutTokenizerProtocol
from .prompt_profiles import (
    InitialContextProfile,
    PromptFragmentKind,
    PublicPromptFragment,
    validate_initial_context_profile,
    validate_prompt_fragments,
)
from .types import DecodingSnapshot, RolloutTask

# The model-facing pass and action contracts are part of persisted H0.  Keep
# their identity disjoint from artifacts rendered before the exact benchmark
# action schemas were made visible.
INITIAL_CONTEXT_FORMAT_VERSION: Final = "ttb-initial-context@6"

# F2/F3 proves that at most five active documents can apply to one task under
# the frozen two-cycle plan.  Every newly authored document must fit this
# complete rendered-block cap at every possible position, including labels and
# immutable metadata.  The limit is applied before a document can enter the
# library; no renderer truncation exists.
AUTHORED_SKILL_FULL_BLOCK_TOKEN_CAP: Final = 3_400
MAXIMUM_APPLICABLE_SKILL_POSITION: Final = 5

_COMPLETION_BENCHMARKS: Final = frozenset(
    {
        "aime-2026",
        "healthbench",
        "hotpotqa",
        "mbpp-plus",
        "triviaqa",
    }
)


def _task_input_profile(task: RolloutTask) -> str:
    # Native IID inputs use the same executor without being relabelled as the
    # historical training source bridge (notably TriviaQA RC vs closed book).
    context = task.public_context
    profile = (
        context.get("input_profile", TRAINING_PUBLIC_INPUT)
        if isinstance(context, dict)
        else TRAINING_PUBLIC_INPUT
    )
    if not isinstance(profile, str):
        raise ValueError("public input profile must be declared text")
    return profile


def _task_benchmark_id(task: RolloutTask) -> str | None:
    context = task.public_context
    if not isinstance(context, dict):
        return None
    benchmark_id = context.get("benchmark_id")
    return benchmark_id if type(benchmark_id) is str and benchmark_id else None


def _benchmark_action_guidance(task: RolloutTask) -> tuple[str, ...]:
    if task.action_surface is not None:
        return _surface_action_guidance(task.action_surface)
    benchmark_id = _task_benchmark_id(task)
    tools = set(task.available_tools)
    if benchmark_id in _COMPLETION_BENCHMARKS or not tools or tools <= {"submit"}:
        return (
            "Submit the final response in arguments.value.answer as a non-empty string:",
            _completion_example({"answer": "YOUR_FINAL_ANSWER"}),
        )
    if benchmark_id == "webshop" or {"click", "purchase", "search"} <= tools:
        return (
            "WebShop accepts only these tool actions:",
            _action_example(
                kind="tool",
                name="search",
                arguments={"query": "SEARCH_QUERY"},
                resource_id="webshop",
            ),
            _action_example(
                kind="tool",
                name="click",
                arguments={"target": "VISIBLE_TARGET"},
                resource_id="webshop",
            ),
            _action_example(
                kind="tool",
                name="purchase",
                arguments={},
                resource_id="webshop",
            ),
            "Use targets from the latest observation. The environment terminates the episode; "
            "do not emit kind=complete.",
        )
    if benchmark_id == "alfworld":
        return (
            "ALFWorld accepts one admissible command per tool action:",
            _action_example(
                kind="tool",
                name="act",
                arguments={"command": "ONE_ADMISSIBLE_COMMAND"},
                resource_id="alfworld",
            ),
            "Choose from admissible_commands in the current public context or latest "
            "observation. The environment terminates the episode; do not emit kind=complete.",
        )
    if "spreadsheet.execute" in tools:
        return (
            "Edit the workbook by sending Python code that uses WORKBOOK_PATH:",
            _action_example(
                kind="tool",
                name="execute",
                arguments={"code": "PYTHON_CODE_EDITING_WORKBOOK_PATH"},
                resource_id="spreadsheet",
            ),
            "After the workbook has been edited and saved, submit this public sentinel:",
            _completion_example({"submit": True}),
        )
    if "appworld.execute" in tools:
        return (
            "Interact with AppWorld by sending one Python program per tool action:",
            _action_example(
                kind="tool",
                name="execute",
                arguments={"code": "PYTHON_CODE_USING_APPWORLD_APIS"},
                resource_id="appworld",
            ),
            "When the requested state changes are complete, submit this public sentinel:",
            _completion_example({"submit": True}),
        )
    return (
        "For a qualified tool identifier resource.name, set resource_id=resource and "
        "name=name; arguments must exactly follow the public task schema. The submit "
        "identifier denotes a kind=complete action, not a tool call.",
    )


def _surface_action_guidance(surface: ActionSurface) -> tuple[str, ...]:
    return ActionContract.freeze(surface).render_public_instruction()


def _step_zero_terminal_wire(task: RolloutTask) -> str | None:
    if not isinstance(task.public_context, dict):
        return None
    value = task.public_context.get("step_zero_terminal_wire")
    return value if type(value) is str and value else None


def _render_action_guidance(
    task: RolloutTask,
    retrieved_skill_ids: tuple[str, ...],
    *,
    profile: InitialContextProfile = InitialContextProfile.TRAINED_SKILLEV,
) -> str:
    terminal_wire = _step_zero_terminal_wire(task)
    if profile is InitialContextProfile.SEEDED_STEP_ZERO and terminal_wire is not None:
        terminal_instructions = {
            "short-answer": "Output one concise final answer line and no explanation.",
            "integer-0-999": "Output exactly one base-10 integer from 0 through 999.",
            "natural-language": "Output the complete benchmark-native natural-language answer.",
            "python-source": (
                "Output only executable Python source, without markdown fences or prose."
            ),
        }
        try:
            instruction = terminal_instructions[terminal_wire]
        except KeyError as exc:
            raise ValueError("unknown Step-0 terminal wire") from exc
        return "\n".join(
            (
                "The frozen benchmark adapter uses a reasoning pass followed by a terminal pass.",
                "Reasoning pass: solve the public task using any helpful approach.",
                f"Terminal pass: {instruction}",
                "These lines specify phase and output shape only; they do not prescribe a "
                "solution method. The terminal payload is not JSON.",
            )
        )
    lines = [
        "The prompt suffix selects one of two passes.",
        "Reasoning pass: solve the task using the public state and any helpful approach.",
        "Action pass: send one JSON action using the interfaces and examples below.",
        "Action wire structured-action-json@3: complete has arguments,kind,name; "
        "tool adds resource_id.",
        f"Declared task tool identifiers: {canonical_json(list(task.available_tools))}",
        *_benchmark_action_guidance(task),
    ]
    exposure = (
        task.public_context.get("skill_exposure_mode")
        if isinstance(task.public_context, dict)
        else None
    )
    if retrieved_skill_ids and exposure != "full-inline-no-invoke":
        lines.extend(
            (
                "A retrieved skill may be invoked only by its visible ID, for example:",
                _action_example(
                    kind="skill",
                    name="invoke",
                    arguments={},
                    resource_id="skill-runtime",
                    skill_id=retrieved_skill_ids[0],
                ),
            )
        )
    return "\n".join(lines)


def _require_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be non-empty text")
    return value


def render_retrieved_skill_block(
    skill: FullRetrievedSkillContext,
    *,
    position: int,
) -> str:
    """Render the one canonical full-content block admitted into ``H_0``."""

    if not isinstance(skill, FullRetrievedSkillContext):
        raise TypeError("retrieved skill block requires FullRetrievedSkillContext")
    if type(position) is not int or position < 1:
        raise ValueError("retrieved skill position must be positive")
    metadata = skill.metadata
    return (
        f"[{position}] id={metadata.skill_id}\n"
        f"version={metadata.version}\n"
        f"content_hash={metadata.content_hash}\n"
        f"content:\n{skill.content}\n"
    )


def retrieved_skill_block_token_count(
    skill: FullRetrievedSkillContext,
    *,
    position: int,
    tokenizer: TokenizerProtocol,
) -> int:
    """Count the exact canonical block, including metadata and separators."""

    return len(tokenizer.encode(render_retrieved_skill_block(skill, position=position)))


def maximum_retrieved_skill_block_token_count(
    skill: FullRetrievedSkillContext,
    *,
    maximum_position: int,
    tokenizer: TokenizerProtocol,
) -> int:
    """Return the worst exact block count over all admitted positions."""

    if type(maximum_position) is not int or maximum_position < 1:
        raise ValueError("maximum retrieved skill position must be positive")
    return max(
        retrieved_skill_block_token_count(skill, position=position, tokenizer=tokenizer)
        for position in range(1, maximum_position + 1)
    )


@dataclass(frozen=True, slots=True)
class AssembledInitialContext:
    """The complete model-visible ``H_0`` payload and its Phase 1 commitment."""

    text: str
    contract: InitialContext

    def __post_init__(self) -> None:
        _require_text(self.text, field="initial context text")
        if not isinstance(self.contract, InitialContext):
            raise ValueError("contract must be an InitialContext")
        if assembled_context_hash(self.text) != self.contract.assembled_hash:
            raise ValueError("initial context text does not match its assembled hash")

    def to_value(self) -> dict[str, JsonValue]:
        """Return the private rollout-artifact representation including full text."""

        return {
            "contract": self.contract.to_value(),
            "text": self.text,
        }

    @classmethod
    def from_value(cls, value: object) -> AssembledInitialContext:
        if not isinstance(value, dict):
            raise ValueError("assembled initial context must be a JSON object")
        normalized = normalize_json(value)
        if not isinstance(normalized, dict) or normalized != value:
            raise ValueError("assembled initial context must be a JSON object")
        if set(normalized) != {"contract", "text"}:
            raise ValueError("assembled initial context has an incompatible field set")
        text = normalized["text"]
        if type(text) is not str:
            raise ValueError("assembled initial context text must be text")
        return cls(
            text=text,
            contract=InitialContext.from_value(normalized["contract"]),
        )


class InitialContextAssembler(Protocol):
    @property
    def assembler_version(self) -> str: ...

    def assemble(
        self,
        *,
        task: RolloutTask,
        retrieved_skills: tuple[FullRetrievedSkillContext, ...],
        active_skill_ids: tuple[str, ...],
        library_version: str,
        tokenizer: RolloutTokenizerProtocol,
        profile: InitialContextProfile = InitialContextProfile.TRAINED_SKILLEV,
        decoding: DecodingSnapshot | None = None,
    ) -> AssembledInitialContext: ...


class CanonicalInitialContextAssembler:
    """Default deterministic, answer-free initial-context assembler."""

    def __init__(
        self,
        *,
        maximum_h0_tokens: int,
        input_window: ModelInputWindow | None = None,
        phase_context: bool = False,
        reasoning_tool_catalog: bool = False,
        token_budget_notice: bool = False,
        action_wire: str = "structured-action-json@3",
        skill_exposure: str = "full-inline",
        public_action_semantics: bool = False,
        task_semantic_guidance: str = LEGACY_TASK_SEMANTICS,
        hotpot_deliberation: bool = False,
    ) -> None:
        if type(maximum_h0_tokens) is not int or maximum_h0_tokens < 1:
            raise ValueError("maximum_h0_tokens must be positive")
        if action_wire not in ACTION_WIRES:
            raise ValueError("unsupported action wire")
        if type(token_budget_notice) is not bool or (token_budget_notice and not phase_context):
            raise ValueError("token budget notice requires an explicit phase context")
        if action_wire != "structured-action-json@3" and not phase_context:
            raise ValueError("native wire requires the phase context candidate")
        if type(reasoning_tool_catalog) is not bool or (
            reasoning_tool_catalog and (not phase_context or action_wire not in NATIVE_TOOL_WIRES)
        ):
            raise ValueError("reasoning tool catalog requires native phase context")
        if skill_exposure not in SKILL_EXPOSURES:
            raise ValueError("unsupported skill exposure condition")
        if type(public_action_semantics) is not bool or (
            public_action_semantics and action_wire not in NATIVE_TOOL_WIRES
        ):
            raise ValueError("public native semantics require the native action wire")
        validate_task_semantic_guidance(task_semantic_guidance)
        if type(hotpot_deliberation) is not bool:
            raise TypeError("Hotpot deliberation must be boolean")
        if task_semantic_guidance != LEGACY_TASK_SEMANTICS and (
            action_wire in NATIVE_TOOL_WIRES and not public_action_semantics
        ):
            raise ValueError("shared native task guidance requires public action semantics")
        self._task_semantic_guidance = task_semantic_guidance
        self._hotpot_deliberation = hotpot_deliberation
        self._public_action_semantics = public_action_semantics
        self._skill_exposure = skill_exposure
        self._phase_context = phase_context
        self._reasoning_tool_catalog = reasoning_tool_catalog
        self._token_budget_notice = token_budget_notice
        self._action_wire = action_wire
        self._maximum_h0_tokens = maximum_h0_tokens
        self._input_window = input_window

    @property
    def skill_exposure(self) -> str:
        return self._skill_exposure

    @property
    def action_wire(self) -> str:
        return self._action_wire

    @property
    def assembler_version(self) -> str:
        return (
            (
                (
                    "phase-context@3/"
                    if self._token_budget_notice
                    else "phase-context@2/"
                    if self._reasoning_tool_catalog
                    else "phase-context@1/"
                )
                + self._action_wire
                if self._phase_context
                else INITIAL_CONTEXT_FORMAT_VERSION
            )
            + ("/" + self._skill_exposure if self._skill_exposure != "full-inline" else "")
            + ("/public-action-semantics@1" if self._public_action_semantics else "")
            + (
                f"/{self._task_semantic_guidance}/hotpot={int(self._hotpot_deliberation)}"
                if self._task_semantic_guidance != LEGACY_TASK_SEMANTICS
                else ""
            )
        )

    def assemble(
        self,
        *,
        task: RolloutTask,
        retrieved_skills: tuple[FullRetrievedSkillContext, ...],
        active_skill_ids: tuple[str, ...],
        library_version: str,
        tokenizer: RolloutTokenizerProtocol,
        profile: InitialContextProfile = InitialContextProfile.TRAINED_SKILLEV,
        decoding: DecodingSnapshot | None = None,
    ) -> AssembledInitialContext:
        if not isinstance(task, RolloutTask):
            raise ValueError("task must be a RolloutTask")
        if self._token_budget_notice and decoding is None:
            raise ValueError("token budget notice requires actual request decoding caps")
        if not isinstance(retrieved_skills, tuple):
            raise ValueError("retrieved_skills must be a tuple")
        if (
            not isinstance(active_skill_ids, tuple)
            or tuple(sorted(set(active_skill_ids))) != active_skill_ids
            or any(type(skill_id) is not str or not skill_id for skill_id in active_skill_ids)
        ):
            raise ValueError("active_skill_ids must be sorted unique non-empty text")
        if not isinstance(library_version, str) or not library_version.strip():
            raise ValueError("library_version must be non-empty text")
        validate_initial_context_profile(
            profile=profile,
            active_skill_ids=active_skill_ids,
            retrieved_skill_ids=tuple(skill.metadata.skill_id for skill in retrieved_skills),
        )
        validate_prompt_fragments(_public_fragments(task, profile=profile), profile=profile)
        if profile is InitialContextProfile.SKILLEV_COLD_START:
            _reject_cold_start_strategy_keys(task.public_context)
        elif profile is InitialContextProfile.SEEDED_STEP_ZERO:
            _reject_seeded_step_zero_authority_keys(task.public_context)

        skill_ids: list[str] = []
        retrieval_inclusions: list[JsonValue] = []
        rendered_skills: list[str] = []
        for position, skill in enumerate(retrieved_skills, start=1):
            metadata = skill.metadata
            if metadata.skill_id in skill_ids:
                raise ValueError("retrieved skill IDs must be unique")
            if metadata.skill_id not in active_skill_ids:
                raise ValueError("retrieved skill is not active in the pinned library")
            skill_ids.append(metadata.skill_id)
            retrieval_inclusions.append(
                {
                    "inclusion_reason": skill.inclusion_reason.value,
                    "skill_id": metadata.skill_id,
                }
            )
            if self._skill_exposure in CATALOG_EXPOSURES:
                from .catalog import render_skill_catalog_entry

                rendered_skills.append(render_skill_catalog_entry(skill, position=position))
            else:
                rendered_skills.append(render_retrieved_skill_block(skill, position=position))

        skill_section = (
            "### Retrieved Skills\n"
            "Optional guidance; adapt it to the task rather than treating it as a required plan.\n"
            + "".join(rendered_skills)
            if rendered_skills
            else ""
        )
        if rendered_skills and self._skill_exposure in CATALOG_EXPOSURES:
            skill_section = (
                "### Available Skill Catalog\nOptional guidance: invoke an exact visible ID "
                "to read its immutable body.\n" + "".join(rendered_skills)
            )
        if self._skill_exposure == TWO_SKILL_CATALOG_EXPOSURE:
            from .catalog import TWO_SKILL_GUIDANCE

            skill_section = (
                "### Available Skill Catalog\n"
                + TWO_SKILL_GUIDANCE
                + "\n"
                + ("".join(rendered_skills) or "No applicable skills are available.\n")
            )
        elif self._skill_exposure == PROACTIVE_CATALOG_EXPOSURE:
            from .catalog import PROACTIVE_SKILL_GUIDANCE

            skill_section = (
                "### Available Skill Catalog\n"
                + PROACTIVE_SKILL_GUIDANCE
                + "\n"
                + ("".join(rendered_skills) or "No applicable skills are available.\n")
            )
        if self._task_semantic_guidance != LEGACY_TASK_SEMANTICS:
            surface = task.action_surface
            benchmark = _task_benchmark_id(task)
            if surface is None or benchmark is None:
                raise ValueError(
                    "shared task guidance requires an explicit public action surface/domain"
                )
            if surface.instructions is None and surface.public_instructions:
                raise ValueError("shared task guidance requires partitioned public instructions")
            semantic = public_task_semantics(
                benchmark,
                input_profile=_task_input_profile(task),
                hotpot_deliberation=self._hotpot_deliberation,
                version=self._task_semantic_guidance,
            )
            instructions: tuple[str, ...] = (semantic,)
            if surface.completion is not None:
                instructions += (TRAINING_SUBMISSION_INSTRUCTION,)
            # Replace only the explicit semantic partition, never task material or
            # prose matched by a heuristic. Old conditions never enter this branch.
            task = replace(
                task,
                action_surface=replace(
                    surface,
                    format=ACTION_SURFACE_FORMAT_V3,
                    public_instructions=(),
                    instructions=PublicActionInstructions(
                        instructions,
                        surface.instructions.json_wire if surface.instructions else (),
                    ),
                ),
            )
        action_guidance = _render_action_guidance(task, tuple(skill_ids), profile=profile)
        action_contract = ActionContract.freeze(
            task.action_surface,
            retrieved_skill_ids=tuple(skill_ids),
            active_skill_ids=active_skill_ids,
        )
        if self._action_wire in NATIVE_TOOL_WIRES:
            action_guidance = (
                "Reasoning and action are separate phases. In the action phase, "
                "emit one native tool call. Optional skills may be read by exact "
                "visible ID; submission need not follow a tool call."
            )
            if self._skill_exposure == TWO_SKILL_CATALOG_EXPOSURE:
                action_guidance = (
                    "Reasoning drafts do not execute actions. In the action phase, call one "
                    "supplied function using its declared parameters. Visible skill IDs are "
                    "values for read_skill's skill_id parameter, not function names. "
                    "Use read_skill for the two-distinct-skills goal in the catalog instructions. "
                    "Each read consumes one action and one turn. "
                )
                if self._action_wire in {NATIVE_TOOL_CARRIER_WIRE, NATIVE_TOOL_HANDOFF_WIRE}:
                    action_guidance += (
                        "The interface accepts a native tool call or an explicit JSON function "
                        "call with name and arguments. A bare argument object is accepted only "
                        "when its fields identify one declared tool unambiguously. "
                        "Surrounding commentary is not executed."
                    )
            elif self._action_wire == NATIVE_TOOL_CARRIER_WIRE:
                action_guidance = (
                    "Use the declared tools to act or submit the final answer. "
                    "The interface accepts a native tool call or a JSON function call "
                    "with name and arguments. A bare argument object is also accepted "
                    "when its fields identify one declared tool unambiguously. "
                    "Surrounding commentary is not executed. Optional skills may be "
                    "read by visible ID; a skill read is not required for submission."
                )
            elif self._action_wire == NATIVE_TOOL_HANDOFF_WIRE:
                action_guidance = (
                    "Reasoning drafts do not execute actions. The action phase calls one "
                    "of the supplied functions using its declared parameters. "
                    "The interface accepts a native tool call or an explicit JSON function "
                    "call with name and arguments. A bare argument object is accepted only "
                    "when its fields identify one declared tool unambiguously. "
                    "Surrounding commentary is not executed. Visible skill IDs are values "
                    "for read_skill's skill_id parameter, not function names; reading a "
                    "skill is optional."
                )
            if self._public_action_semantics:
                action_guidance = "\n".join(
                    (*action_contract.render_native_semantics(), action_guidance)
                )
            if self._skill_exposure == PROACTIVE_CATALOG_EXPOSURE:
                action_guidance += (
                    " Prefer read_skill when a catalog method is applicable, then use the "
                    "returned advice in your own task actions. Do not make irrelevant calls "
                    "or treat reading as completed verification."
                )
        displayed_query = task.query
        text = (
            f"### Query\n{displayed_query}\n"
            f"{_render_source_messages(task)}"
            f"{skill_section}"
            f"### Public Task Context\n{canonical_json(task.public_context)}\n"
            f"### Available Actions\n{action_guidance}\n"
        )
        if self._phase_context:
            from .catalog import PROACTIVE_SKILL_TOOL_DESCRIPTION

            typed_alfworld = (
                self._task_semantic_guidance
                in {
                    PUBLIC_TASK_SEMANTICS_V2,
                    PUBLIC_TASK_SEMANTICS_V3,
                    PUBLIC_TASK_SEMANTICS_V4,
                    PUBLIC_TASK_SEMANTICS_V5,
                    PUBLIC_TASK_SEMANTICS_V7,
                    PUBLIC_TASK_SEMANTICS_V8,
                    PUBLIC_TASK_SEMANTICS_V9,
                    PUBLIC_TASK_SEMANTICS_V10,
                }
                and _task_benchmark_id(task) == "alfworld"
            )
            public_state = (
                task.public_context.get("payload", {})
                if typed_alfworld and isinstance(task.public_context, dict)
                else None
            )
            repaired = self._task_semantic_guidance in {
                PUBLIC_TASK_SEMANTICS_V8,
                PUBLIC_TASK_SEMANTICS_V9,
                PUBLIC_TASK_SEMANTICS_V10,
            }
            if (
                repaired
                and isinstance(public_state, dict)
                and isinstance(task.public_context, dict)
            ):
                native_limit = task.public_context.get("max_steps")
                if type(native_limit) is int:
                    public_state = {**public_state, "max_steps": native_limit}
            spec = PhaseContextSpec(
                submission_feedback=repaired,
                skill_advice=repaired,
                action_wire=self._action_wire,
                deliverable=phase_deliverable(
                    _task_benchmark_id(task), version=self._task_semantic_guidance
                ),
                reasoning_tool_catalog=self._reasoning_tool_catalog,
                reasoning_token_cap=decoding.max_reasoning_tokens
                if self._token_budget_notice and decoding is not None
                else None,
                action_token_cap=decoding.max_action_tokens
                if self._token_budget_notice and decoding is not None
                else None,
                tools_json=canonical_json(
                    list(
                        action_contract.to_native_tools(
                            public_action_semantics=self._public_action_semantics,
                            skill_read_description=(
                                PROACTIVE_SKILL_TOOL_DESCRIPTION
                                if self._skill_exposure == PROACTIVE_CATALOG_EXPOSURE
                                else "Read one visible skill by exact ID. This trajectory "
                                "should read "
                                "at least two different skills; each read consumes an action/turn."
                                if self._skill_exposure == TWO_SKILL_CATALOG_EXPOSURE
                                else "Read an optional advisory document by exact ID. Reading does "
                                "not execute a task action, check the environment, "
                                "or certify success. "
                                "Each read still consumes a controller turn and tool-call budget."
                                if repaired
                                else None
                            ),
                        )
                    )
                )
                if self._action_wire in NATIVE_TOOL_WIRES
                else "[]",
                action_contract_json=canonical_json(action_contract.to_scoring_metadata()),
                history_format=(OBSERVED_PUBLIC_HISTORY if repaired else TYPED_PUBLIC_HISTORY)
                if typed_alfworld
                else None,
                initial_public_state_json=canonical_json(public_state),
                max_turns=task.budget_profile.max_turns
                if typed_alfworld and task.budget_profile is not None
                else None,
            )
            text = spec.wrap(text)
        token_count = len(encode_rollout_prompt(tokenizer, text))
        if token_count <= 0:
            raise ValueError("assembled initial context must encode to at least one token")
        if token_count > self._maximum_h0_tokens and self._input_window is None:
            raise ValueError("canonical initial context exceeds maximum_h0_tokens")
        meta: dict[str, JsonValue] = {
            "environment_id": task.environment_id,
            "context_id": task.context_id,
            "available_tools": list(task.available_tools),
            "format_version": INITIAL_CONTEXT_FORMAT_VERSION,
            "library_version": library_version,
            "retrieval_inclusions": retrieval_inclusions,
            "root_query_source": "rollout-task-query",
            "root_query_stable_across_episode": True,
            "root_query_token_count": len(tokenizer.encode(task.query)),
            "task_family": task.task_family,
            "task_id": task.task_id,
        }
        if isinstance(task.public_context, dict):
            reset_identity = task.public_context.get("environment_reset_identity")
            if isinstance(reset_identity, str) and reset_identity:
                meta["environment_reset_identity"] = reset_identity
        if self._task_semantic_guidance != LEGACY_TASK_SEMANTICS:
            meta["task_semantic_guidance"] = self._task_semantic_guidance
            meta["task_semantic_input_profile"] = _task_input_profile(task)
            meta["hotpot_deliberation"] = self._hotpot_deliberation
        if self._public_action_semantics:
            meta["public_action_semantics"] = "public-action-semantics@1"
        if self._phase_context or self._skill_exposure != "full-inline":
            meta.update(
                {
                    "action_wire": self._action_wire,
                    "action_contract": action_contract.to_scoring_metadata(),
                    "skill_exposure": self._skill_exposure,
                }
            )
        contract = InitialContext(
            query=task.query,
            retrieved_skill_ids=tuple(skill_ids),
            active_skill_ids=active_skill_ids,
            meta={
                **meta,
                **(
                    {INPUT_WINDOW_META_KEY: self._input_window.to_value()}
                    if self._input_window
                    else {}
                ),
            },
            assembler_version=self.assembler_version,
            assembled_hash=assembled_context_hash(text),
            assembled_token_count=token_count,
        )
        return AssembledInitialContext(text=text, contract=contract)


def _public_fragments(
    task: RolloutTask,
    *,
    profile: InitialContextProfile,
) -> tuple[PublicPromptFragment, ...]:
    fragments = [PublicPromptFragment(PromptFragmentKind.TASK, "task.query", task.query)]
    if task.public_context not in ({}, None):
        fragments.append(
            PublicPromptFragment(
                PromptFragmentKind.PUBLIC_OBSERVATION,
                "task.public_context",
                canonical_json(task.public_context),
            )
        )
    fragments.append(
        PublicPromptFragment(
            PromptFragmentKind.ACTION_SCHEMA,
            "task.action_surface",
            _render_action_guidance(task, (), profile=profile),
        )
    )
    if task.budget_profile is not None:
        fragments.append(
            PublicPromptFragment(
                PromptFragmentKind.BUDGET,
                "task.budget_profile",
                canonical_json(task.budget_profile.to_value()),
            )
        )
    return tuple(fragments)


_COLD_START_FORBIDDEN_KEYS = frozenset(
    {"demonstration", "demonstrations", "strategy", "gold", "gold_derived", "oracle"}
)

_SEEDED_STEP_ZERO_FORBIDDEN_KEYS = frozenset(
    {
        "answer",
        "answer_key",
        "answers",
        "correct_answer",
        "demonstration",
        "demonstrations",
        "expected_answer",
        "gold",
        "gold_derived",
        "label",
        "labels",
        "oracle",
        "reference_answer",
        "reward",
        "rewards",
        "rubric",
        "rubrics",
        "score",
        "scorer",
        "solution",
        "strategy",
        "target_answer",
    }
)


def _reject_cold_start_strategy_keys(value: JsonValue, *, location: str = "public_context") -> None:
    if isinstance(value, dict):
        forbidden = {key.casefold() for key in value}.intersection(_COLD_START_FORBIDDEN_KEYS)
        if forbidden:
            raise ValueError(f"cold-start H0 contains strategy/gold keys at {location}")
        for key, child in value.items():
            _reject_cold_start_strategy_keys(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_cold_start_strategy_keys(child, location=f"{location}[{index}]")


def _reject_seeded_step_zero_authority_keys(
    value: JsonValue,
    *,
    location: str = "public_context",
) -> None:
    """Keep answers, evaluator authority, and ad-hoc strategy outside Step-0 H0.

    Injected skill advice belongs in the typed retrieved-skill block; the model
    remains free to devise its own strategy without a skill. Benchmark
    task text and public observations remain untouched, so this guard cannot
    censor legitimate question wording.
    """

    if isinstance(value, dict):
        forbidden = {key.casefold() for key in value}.intersection(_SEEDED_STEP_ZERO_FORBIDDEN_KEYS)
        if forbidden:
            raise ValueError(f"seeded Step-0 H0 contains forbidden authority at {location}")
        for key, child in value.items():
            _reject_seeded_step_zero_authority_keys(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_seeded_step_zero_authority_keys(child, location=f"{location}[{index}]")


def _render_source_messages(task: RolloutTask) -> str:
    value = [message.to_value() for message in task.model_visible_messages]
    return ROLLOUT_SOURCE_MESSAGES_BEGIN + canonical_json(value) + ROLLOUT_SOURCE_MESSAGES_END
