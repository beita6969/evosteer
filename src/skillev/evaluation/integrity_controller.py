"""Single-owner Step-0 controller with optional text skills and native tools."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace

from skillev.evolution.retriever import TaskRetrievalFeatures
from skillev.rollout import GenerationPhase
from skillev.rollout.evaluation_sglang import EvaluationRolloutGenerator
from skillev.runtime import SkillLibraryState

from .agent_communication import (
    control_failure_feedback,
    control_payload,
)
from .capability_registry import (
    native_tool_definitions,
    runtime_capabilities,
)
from .decision_transport import (
    ExplicitDecision,
    PublicSurface,
    normalize_decision,
    public_repair_feedback,
)
from .direct_baseline import DirectGenerationError, DirectGenerationRequest, DirectGenerationResult
from .direct_baseline.interactive_tasks import (
    NativeEnvironmentOutcome,
    NativeEnvironmentStep,
    NativeInteractiveTask,
    NativePublicState,
)
from .integrity_context import EpisodeContext
from .integrity_generation import IntegrityGeneration
from .integrity_skill_context import IntegritySkillContext
from .interactive_termination import OWNER_FINISH_INSTRUCTION, owner_finish_requested
from .native_tool_instructions import native_tool_instructions
from .owner_final import FinalSubmission, project_owner_final
from .public_context import ArchiveKind, PromptBlock, PublicPrompt, PublicView
from .scienceworld_commands import TYPED_COMMAND_PROFILES, command_profile
from .sealed_candidates import EventRecorder
from .skill_library_config import FrozenSkillLibrary, initial_skill_library
from .step0_completion import (
    StepZeroTerminalMode,
    native_action_constraint,
    terminal_submission_feedback,
)
from .step0_integrity import (
    AgentTopology,
    InterventionCounts,
    ReasoningPass,
    SkillMode,
    ToolCallMode,
)
from .step0_interactive_agent import ArchitectureEpisodeArtifact
from .step0_public_state import ExecutionStatus, PublicEpisodeState, TaskStatus
from .step0_receipts import ArchitectureIdentityReceipt
from .step0_types import (
    StepZeroActionMode,
    StepZeroArchitectureConfig,
    StepZeroCompletionMode,
    StepZeroDiagnostics,
    StepZeroTaskBinding,
)


@dataclass(slots=True)
class InteractiveRuntime:
    memory: PublicEpisodeState
    surface: PublicSurface
    turns: int = 0
    maximum_environment_steps: int | None = None


class IntegrityStepZeroClient:
    def __init__(
        self,
        *,
        generator: EvaluationRolloutGenerator,
        library_state: SkillLibraryState | None,
        bindings: tuple[StepZeroTaskBinding, ...],
        config: StepZeroArchitectureConfig,
        skill_library: FrozenSkillLibrary | None = None,
        journal: EventRecorder | None = None,
        run_id: str = "unpublished-development",
        corpus_search: Callable[[str], str] | None = None,
    ) -> None:
        # OFF never constructs a retriever, reads a body, or injects an empty skill header.
        if config.arm.legacy:
            raise ValueError("clean controller cannot run a historical arm")
        if (
            config.arm.reasoning_pass is ReasoningPass.REQUIRED
            and config.arm.agent_topology is not AgentTopology.TWO_PASS
        ):
            raise ValueError("an additional required pass must be identified as two-pass-single")
        config.arm.validate_live_topology()
        self._config = config
        self._corpus_search = corpus_search
        self._bindings: dict[str, StepZeroTaskBinding] = {}
        self._skill_library: FrozenSkillLibrary | None = None
        if config.arm.skill_mode is not SkillMode.OFF:
            if config.arm.skill_mode is SkillMode.LIBRARY and skill_library is None:
                raise ValueError("explicit library arm requires its restored skill snapshot")
            self._skill_library = skill_library or (
                FrozenSkillLibrary("provided-initial-library", "initial", library_state)
                if library_state is not None
                else initial_skill_library()
            )
            if config.arm.skill_mode is SkillMode.LIBRARY and (
                config.arm.skill_library_id != self._skill_library.library_id
                or config.arm.skill_retrieval_rule != self._skill_library.retrieval_rule
            ):
                raise ValueError("restored skill snapshot differs from the selected arm")
        self._skill_contexts: dict[str, IntegritySkillContext] = {}
        self._interactive: dict[str, InteractiveRuntime] = {}
        self._cached: dict[str, DirectGenerationResult] = {}
        self._reasoning_requested: set[str] = set()
        self._invalid = 0
        self._contexts: dict[str, EpisodeContext] = {}
        self._submissions: dict[str, FinalSubmission] = {}
        self._decisions: dict[str, ExplicitDecision] = {}
        self.counts = InterventionCounts()
        self.generation = IntegrityGeneration(generator, config.arm, self.counts, journal, run_id)
        for binding in bindings:
            self.register_binding(binding)

    def register_binding(self, binding: StepZeroTaskBinding) -> None:
        previous = self._bindings.get(binding.task_id)
        if previous is not None and previous != binding:
            raise ValueError("task route cannot change after registration")
        self._bindings[binding.task_id] = binding

    def request_reasoning(self, task_id: str) -> None:
        """Public controller request, not a benchmark- or score-triggered regeneration."""
        if self._config.arm.reasoning_pass is not ReasoningPass.OPTIONAL:
            raise ValueError("this arm does not permit optional reasoning")
        if task_id not in self._bindings:
            raise KeyError(task_id)
        self._reasoning_requested.add(task_id)

    @property
    def diagnostics(self) -> StepZeroDiagnostics:
        return StepZeroDiagnostics(
            self.counts.model_calls, 0, self._invalid, 0, 0, self._retrieved_ids()
        )

    @property
    def architecture_identity(self) -> ArchitectureIdentityReceipt:
        state, arm = self._config.inference_state, self._config.arm
        return ArchitectureIdentityReceipt(
            method_id=state.method_id,
            architecture_family="skillev-bayesian-improve",
            architecture_lineage="skillflow-plus-idea-tex",
            architecture_components=(
                "seeded-skill-controller",
                "trajectory-balance-gflownet",
                "beta-bernoulli-lcb-calibration",
                "operator-driven-skill-evolution",
            ),
            controller_id=arm.arm_id,
            optimizer_steps=state.optimizer_steps,
            forward_adapter_active=state.forward_adapter_active,
            backward_adapter_active=state.backward_adapter_active,
            posterior_active=state.posterior_active,
            calibration_active=state.calibration_active,
            operator_active=state.operator_active,
            seed_library_id=self._skill_library.library_id if self._skill_library else "no-skills",
            seed_skill_ids=self._retrieved_ids(),
            retrieval_policy_id=self._skill_library.retrieval_rule
            if self._skill_library
            else "off",
            initial_context_profile="public-task-and-events@1",
            reasoning_authority="trained-forward-policy-configured"
            if state.optimizer_steps
            else "frozen-evaluation-configured",
            reasoning_contract_resolved=True,
            action_policy_authority="evaluated-policy-single-final-no-selection@1",
            inference_arm=arm.to_value(),
            intervention_counts=asdict(self.counts),
        )

    def _binding(self, request_id: str) -> tuple[StepZeroTaskBinding, int]:
        if request_id in self._bindings:
            return self._bindings[request_id], 0
        matches = []
        for task_id, binding in self._bindings.items():
            for marker in (":step:", ":search-turn:"):
                suffix = request_id.removeprefix(task_id + marker)
                if suffix != request_id and suffix.isdecimal() and int(suffix) > 0:
                    matches.append((len(task_id), binding, int(suffix)))
        if not matches:
            raise DirectGenerationError("request has no registered public task")
        _, binding, turn = max(matches, key=lambda item: item[0])
        return binding, turn

    def _retrieved_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    skill_id
                    for context in self._skill_contexts.values()
                    for skill_id in context.retrieved
                }
            )
        )

    def _skill_context(self, binding: StepZeroTaskBinding) -> IntegritySkillContext | None:
        if self._skill_library is None:
            return None
        if binding.task_id not in self._skill_contexts:
            self._skill_contexts[binding.task_id] = IntegritySkillContext(
                self._skill_library,
                TaskRetrievalFeatures(
                    task_id=binding.task_id,
                    task_family=binding.task_family,
                    context=binding.context_id or f"{binding.benchmark_id}:task",
                    available_tools=tuple(
                        sorted(
                            entry.capability_id
                            for entry in runtime_capabilities(binding.action_mode.value).entries
                        )
                    ),
                    benchmark_id=binding.benchmark_id,
                    root_query=self._contexts[binding.task_id].root_task,
                ),
                binding.skill_instruction_token_budget,
                self.generation,
            )
        return self._skill_contexts[binding.task_id]

    def _remaining_calls(self, calls_started: int) -> int:
        remaining = self._config.max_turns - (self.counts.model_calls - calls_started)
        if self.generation.call_limit is not None:
            remaining = min(remaining, self.generation.call_limit - self.counts.model_calls)
        return remaining

    def public_view(self, task_id: str) -> PublicView:
        context = self._contexts[task_id]
        runtime = self._interactive.get(task_id)
        if runtime is None:
            return PublicView(0, context.root_task, "", ())
        return PublicView(
            runtime.surface.revision,
            context.root_task,
            runtime.memory.history[-1].observation,
            runtime.surface.native_actions,
            (context.latest_environment_id,) if context.latest_environment_id else (),
        )

    def current_decision(self, task_id: str) -> ExplicitDecision:
        return self._decisions[task_id]

    def final_submission(self, task_id: str) -> FinalSubmission | None:
        return self._submissions.get(task_id)

    def _remember(self, task_id: str, role: str, content: str, kind: ArchiveKind) -> None:
        view = self.public_view(task_id)
        self._contexts[task_id].remember(role, content, view.revision, kind)
        runtime = self._interactive.get(task_id)
        if runtime is not None and kind is ArchiveKind.DISCUSSION:
            runtime.memory = runtime.memory.remember(
                "owner" if role == "assistant" else role, content
            )

    def _owner_prompt(self, binding: StepZeroTaskBinding) -> PublicPrompt:
        context = self._contexts[binding.task_id]
        runtime = self._interactive.get(binding.task_id)
        submission = (
            "Only you submit environment actions. The simulator receives your decisions "
            "through native tools, not a chat answer. "
            if runtime is not None
            else "Only you write and submit the next reply to the user. "
            if binding.completion_mode is StepZeroCompletionMode.NATURAL_LANGUAGE
            else "Only you submit the final answer. "
        )
        capabilities = (
            "Available functions are described in the tool definitions."
            if self._config.arm.tool_call_mode is ToolCallMode.QWEN_XML
            else runtime_capabilities(
                binding.action_mode.value,
                skills=self._skill_library is not None,
                scienceworld_profile=command_profile(self._config.arm.task_semantic_guidance),
            ).render()
        )
        prompt = context.prompt(
            maximum_input_tokens=binding.maximum_h0_tokens or self._config.maximum_h0_tokens
        ).with_instruction(
            capabilities
            + "\nYou may use these capabilities or solve the task yourself. "
            + submission
        )
        if runtime is not None:
            content = (
                "Available environment functions: act(command), inspect_object(target), "
                "select_task_object(target). act accepts free-form native commands. "
                "inspect_object sends look at; select_task_object sends focus on. "
                "All use your literal arguments. Submit one function call or one final "
                "Action: command field."
                if runtime.surface.mode == "scienceworld"
                and command_profile(self._config.arm.task_semantic_guidance)
                in TYPED_COMMAND_PROFILES
                else "Available environment function: act(command). Its command argument "
                "is free-form ScienceWorld text, not a choice from an action menu. "
                "Submit one explicit act call or one final Action: command field."
                if runtime.surface.mode == "scienceworld"
                else "Available actions (complete public surface):\n"
                + json.dumps(runtime.surface.native_actions, ensure_ascii=False)
            )
            if runtime.maximum_environment_steps is not None:
                content += (
                    "\nRemaining environment actions: "
                    f"{runtime.maximum_environment_steps - runtime.turns}."
                )
            if binding.owner_finish_allowed:
                content += "\n" + OWNER_FINISH_INSTRUCTION
            prompt = prompt.append(PromptBlock("tool", content, runtime_metadata=True))
        skills = self._skill_context(binding)
        if skills is not None:
            for block in skills.blocks.values():
                prompt = prompt.append(block)
        return prompt

    def _history_read(self, task_id: str, internal: dict[str, object]) -> str:
        context = self._contexts[task_id]
        if "archive" in internal:
            kind = ArchiveKind(str(internal["archive"]))
            cursor, limit = internal.get("cursor", 0), internal.get("limit", 4)
            if type(cursor) is not int or type(limit) is not int:
                raise ValueError("history cursor and limit must be integers")
            value = context.history_page(kind, cursor=cursor, limit=limit)
        else:
            # Named compatibility for the previous revision range interface.
            first, last = internal.get("first"), internal.get("last")
            if type(first) is not int or type(last) is not int or not 1 <= first <= last:
                raise ValueError("history range is invalid")
            value = context.history_page(
                ArchiveKind.ENVIRONMENT, cursor=first - 1, limit=min(8, last - first + 1)
            )
        self.generation.trace(task_id, "history-read", {"request": internal, "page": value})
        self.counts.tool_calls += 1
        return json.dumps(value, ensure_ascii=False)

    async def generate(self, request: DirectGenerationRequest) -> DirectGenerationResult:
        if request.request_id in self._cached:
            return self._cached[request.request_id]
        binding, _turn = self._binding(request.request_id)
        task_id, arm = binding.task_id, self._config.arm
        runtime = self._interactive.get(task_id)
        direct_submission = (
            "one act(command), inspect_object(target), or select_task_object(target) call, "
            "or one Action: command field"
            + (", or finish() to end the episode" if binding.owner_finish_allowed else "")
            if binding.action_mode is StepZeroActionMode.SCIENCE_WORLD
            and command_profile(arm.task_semantic_guidance) in TYPED_COMMAND_PROFILES
            else "one act(command) call, an Action: command field, or finish() to end the episode"
            if binding.owner_finish_allowed
            else "one act(command) call or an Action: command field"
            if binding.action_mode is StepZeroActionMode.SCIENCE_WORLD
            else "a native action"
            if runtime is not None
            else "your next reply to the user"
            if binding.completion_mode is StepZeroCompletionMode.NATURAL_LANGUAGE
            else "your final answer"
        )
        if (
            binding.action_mode
            in {
                StepZeroActionMode.WEB_SHOP,
                StepZeroActionMode.ALF_WORLD,
                StepZeroActionMode.SCIENCE_WORLD,
            }
            and runtime is None
        ):
            raise DirectGenerationError("interactive public state is not initialized")
        if task_id not in self._contexts:
            root_task = "\n".join(
                item["content"] for item in request.messages if item["role"] != "system"
            )
            self._contexts[task_id] = EpisodeContext(request.messages, root_task)
        self.generation.trace(
            task_id, "input", {"request_id": request.request_id, "messages": request.messages}
        )
        tools = (
            native_tool_definitions(
                binding.action_mode.value,
                skills=self._skill_library is not None,
                natural_language=binding.completion_mode is StepZeroCompletionMode.NATURAL_LANGUAGE,
                owner_finish=binding.owner_finish_allowed,
                scienceworld_profile=command_profile(arm.task_semantic_guidance),
                corpus_search=self._corpus_search is not None,
            )
            if arm.tool_call_mode is ToolCallMode.QWEN_XML
            else ()
        )
        generator = self.generation.generator
        episode_id = f"integrity:{request.request_id}"
        calls_started = self.counts.model_calls
        prompt_tokens = completion_tokens = 0
        thoughts: list[str] = []
        final, finish_reason = "", "budget-exhausted"
        unresolved: list[str] = []
        repair_reason: str | None = None
        repair_in_progress = False
        repair_call_id = ""

        def repair_result(success: bool) -> None:
            nonlocal repair_in_progress
            if repair_in_progress:
                self.generation.trace(
                    task_id,
                    "model-interface-repair",
                    {
                        "status": "completed",
                        "success": success,
                        "call_id": repair_call_id,
                    },
                )
                repair_in_progress = False

        def feedback(text: str, reason: str) -> None:
            nonlocal repair_reason
            repair_result(False)
            if runtime is not None:
                self._contexts[task_id].interface_feedback(text, self.public_view(task_id).revision)
            else:
                self._remember(task_id, "tool", text, ArchiveKind.TOOL_RESULT)
            repair_reason = reason

        def resolve_controls(resolution: str, current_attempt: str | None = None) -> None:
            repair_result(True)
            self._contexts[task_id].resolve_interface_feedback()
            for attempt_id in dict.fromkeys(
                (*unresolved, *((current_attempt,) if current_attempt else ()))
            ):
                self.generation.trace(
                    task_id,
                    "control-resolved",
                    {"attempt_id": attempt_id, "resolution": resolution},
                )
            unresolved.clear()

        generator.begin_episode(episode_id, generator.snapshot().snapshot_id)
        try:
            reasoning = (
                arm.reasoning_pass is ReasoningPass.REQUIRED or task_id in self._reasoning_requested
            )
            self._reasoning_requested.discard(task_id)
            if reasoning:
                text, thought, generated = await self.generation.call(
                    request,
                    task_id,
                    self._owner_prompt(binding).with_instruction(
                        "Work on the task; your analysis will be available "
                        "for your final response or next action."
                    ),
                    phase=GenerationPhase.REASONING,
                    maximum_tokens=binding.max_reasoning_tokens
                    or self._config.max_reasoning_tokens,
                    purpose="optional-review"
                    if arm.reasoning_pass is ReasoningPass.OPTIONAL
                    else "declared-two-pass",
                    public_view=self.public_view(task_id),
                )
                thoughts.extend((thought, text))
                prompt_tokens += generated.usage.input_tokens
                completion_tokens += generated.usage.output_tokens
                self._remember(task_id, "assistant", text or thought, ArchiveKind.DISCUSSION)
            while self._remaining_calls(calls_started) > 0:
                decided_on = self.public_view(task_id)
                if runtime is not None:
                    contract = native_action_constraint(
                        binding.action_mode.value, available_actions=decided_on.actions
                    )
                    self.generation.trace(
                        task_id,
                        "action-surface",
                        {
                            "mode": runtime.surface.mode,
                            "revision": decided_on.revision,
                            "native_actions": decided_on.actions,
                            "advertised_actions": decided_on.actions,
                            "tool_schema": contract.json_schema,
                            "generation_grammar_applied": False,
                        },
                    )
                repair_in_progress = repair_reason is not None
                text, thought, generated = await self.generation.call(
                    request,
                    task_id,
                    self._owner_prompt(binding)
                    .with_instruction(f"You can submit {direct_submission} directly.")
                    .with_runtime_metadata(
                        "Calls remaining to prepare this one environment action: "
                        f"{self._remaining_calls(calls_started)}. This is a local "
                        "response/repair allowance, NOT the remaining environment "
                        "actions. After an action is delivered, a new local allowance "
                        "is available for the next action, subject to the remaining "
                        "episode limits. The environment-action, total model-call "
                        "and total output-token budgets do not reset."
                        if runtime is not None
                        else "Calls remaining for this response or environment action: "
                        f"{self._remaining_calls(calls_started)}."
                    ),
                    phase=GenerationPhase.ACTION,
                    maximum_tokens=binding.max_action_tokens or self._config.max_action_tokens,
                    maximum_calls=self._remaining_calls(calls_started),
                    purpose="interface-repair" if repair_reason is not None else "decision",
                    public_view=decided_on,
                    tools=tools,
                )
                repair_reason = None
                repair_call_id = self.generation.calls[-1].call_id
                finish_reason = generated.finish_reason
                thoughts.append(thought)
                prompt_tokens += generated.usage.input_tokens
                completion_tokens += generated.usage.output_tokens
                if arm.native_thinking and thought:
                    self._contexts[task_id].remember_reasoning(thought, decided_on.revision)
                self._remember(task_id, "assistant", text, ArchiveKind.DISCUSSION)
                if binding.owner_finish_allowed and owner_finish_requested(text):
                    finish_reason = "owner-finish"
                    resolve_controls("owner-finish")
                    break
                attempt_id = f"control-{self.counts.model_calls}"
                try:
                    internal = control_payload(text)
                except ValueError:
                    self._invalid += 1
                    unresolved.append(attempt_id)
                    self.generation.trace(task_id, "control-attempt", {"attempt_id": attempt_id})
                    self.generation.trace(
                        task_id,
                        "control-parse-failure",
                        {"attempt_id": attempt_id, "message_delivered": False},
                    )
                    feedback(
                        control_failure_feedback(
                            text,
                            direct_submission=direct_submission,
                            finish_reason=generated.finish_reason,
                            action_mode=binding.action_mode.value if runtime is not None else None,
                        ),
                        "control-encoding",
                    )
                    continue
                if internal is not None and internal.get("kind") in {
                    "message",
                    "review",
                    "history",
                    "skill",
                    "corpus_search",
                }:
                    self.generation.trace(
                        task_id,
                        "control-attempt",
                        {"attempt_id": attempt_id, "kind": internal["kind"]},
                    )
                    self.generation.trace(
                        task_id,
                        "control-decoded",
                        {"attempt_id": attempt_id, "kind": internal["kind"]},
                    )
                if internal is not None and internal.get("kind") == "corpus_search":
                    if self._corpus_search is None:
                        unresolved.append(attempt_id)
                        feedback(
                            "Corpus search is not available in this condition.",
                            "corpus-unavailable",
                        )
                    else:
                        queries = internal.get("queries", [internal.get("query")])
                        for query in queries:
                            self.counts.tool_calls += 1
                            corpus_result = self._corpus_search(str(query))
                            self._remember(task_id, "tool", corpus_result, ArchiveKind.TOOL_RESULT)
                        resolve_controls("corpus-result-delivered", attempt_id)
                    continue
                if internal is not None and internal.get("kind") == "skill":
                    skills = self._skill_context(binding)
                    try:
                        if skills is None:
                            raise ValueError(
                                "skill access is disabled; basic tools remain available"
                            )
                        skill_result = skills.execute(internal)
                    except ValueError as error:
                        unresolved.append(attempt_id)
                        feedback(f"Skill request not executed: {error}", "skill-request")
                    else:
                        self._remember(
                            task_id,
                            "tool",
                            json.dumps(skill_result, ensure_ascii=False),
                            ArchiveKind.TOOL_RESULT,
                        )
                        resolve_controls("skill-result-delivered", attempt_id)
                    continue
                if internal is not None and internal.get("kind") == "review":
                    if not isinstance(internal.get("body"), str) or not internal["body"].strip():
                        unresolved.append(attempt_id)
                        feedback(
                            "Review was not recorded: its body is empty or not text.", "review-body"
                        )
                    else:
                        self._remember(
                            task_id,
                            "tool",
                            "Review recorded as model discussion; no environment tool was invoked. "
                            "Continue when ready.",
                            ArchiveKind.TOOL_RESULT,
                        )
                        resolve_controls("model-self-edit", attempt_id)
                    continue
                if internal is not None and internal.get("kind") == "history":
                    try:
                        history = self._history_read(task_id, internal)
                    except ValueError:
                        unresolved.append(attempt_id)
                        feedback(
                            "The archive request was not executed. Use a listed archive name, "
                            "nonnegative cursor and page limit from 1 through 8.",
                            "history-request",
                        )
                    else:
                        self._remember(task_id, "tool", history, ArchiveKind.TOOL_RESULT)
                        resolve_controls("history-delivered", attempt_id)
                    continue
                if internal is not None and internal.get("kind") == "message":
                    self._invalid += 1
                    unresolved.append(attempt_id)
                    self.generation.trace(
                        task_id,
                        "peer-delivery-failure",
                        {
                            "attempt_id": attempt_id,
                            "message_delivered": False,
                            "reason": "single-owner-topology",
                        },
                    )
                    feedback(
                        "No other agents are available in this episode; no message was sent. "
                        f"Use your available tools or submit {direct_submission} directly.",
                        "unsupported-agent-request",
                    )
                    continue
                if runtime is None:
                    mode = _terminal_mode(binding)
                    submission = (
                        project_owner_final(
                            mode, text, owner_id="owner", message_id=f"{task_id}:owner-final"
                        )
                        if mode
                        else None
                    )
                    if mode and binding.action_mode is not StepZeroActionMode.TRIVIA_SEARCH:
                        if submission is None:
                            self._invalid += 1
                            self.generation.trace(
                                task_id,
                                "terminal-parse-failure",
                                {"mode": mode.value, "candidate_submitted": False},
                            )
                            feedback(terminal_submission_feedback(mode), "terminal-payload")
                            continue
                        final = submission.payload
                        self._submissions[task_id] = submission
                        self.generation.trace(task_id, "owner-final-submission", asdict(submission))
                    else:
                        final = text
                    resolve_controls("owner-direct-final")
                    break
                decision = ExplicitDecision(text, decided_on.revision)
                transported = normalize_decision(decision, runtime.surface)
                self.generation.trace(
                    task_id,
                    "decision-transport",
                    {
                        "decision": asdict(decision),
                        "surface": asdict(runtime.surface),
                        "result": asdict(transported),
                    },
                )
                if transported.action is not None:
                    self._decisions[task_id] = decision
                    final = "Action: " + transported.action
                    resolve_controls("owner-direct-action")
                    break
                self._invalid += 1
                feedback(
                    public_repair_feedback(
                        transported, runtime.surface, finish_reason=generated.finish_reason
                    )
                    + ("\n" + OWNER_FINISH_INSTRUCTION if binding.owner_finish_allowed else ""),
                    "native-action",
                )
            if (
                not final
                and finish_reason != "owner-finish"
                and self._remaining_calls(calls_started) <= 0
            ):
                finish_reason = "budget-exhausted"
            self.counts.require_arm(arm)
            if runtime is not None:
                runtime.turns += 1
                runtime.memory = replace(
                    runtime.memory, model_notes="\n".join(item for item in thoughts if item)
                )
            result = DirectGenerationResult(
                request.request_id,
                final,
                finish_reason,
                prompt_tokens,
                completion_tokens,
                "\n".join(thoughts).strip() or None,
                self._config.response_model,
                f"{request.request_id}:policy-final",
                self._config.service_instance_id,
            )
            self._cached[request.request_id] = result
            return result
        finally:
            generator.end_episode(episode_id)

    async def begin_interactive_episode(
        self,
        task: NativeInteractiveTask,
        state: NativePublicState,
        *,
        semantic_instruction: str = "",
    ) -> None:
        if task.task_id in self._interactive:
            raise ValueError("episode already active")
        memory = PublicEpisodeState(
            task.task,
            history_window_steps=task.history_window_steps,
            history_maximum_characters=task.history_maximum_characters,
        ).observe(None, state.observation_text)
        binding = self._bindings[task.task_id]
        self._interactive[task.task_id] = InteractiveRuntime(
            memory,
            PublicSurface(binding.action_mode.value, memory.revision, state.available_actions),
            maximum_environment_steps=task.max_steps,
        )
        context = EpisodeContext(
            (
                {
                    "role": "system",
                    "content": (semantic_instruction + "\n" if semantic_instruction else "")
                    + native_tool_instructions(
                        binding.action_mode.value,
                        native_functions=self._config.arm.tool_call_mode is ToolCallMode.QWEN_XML,
                        scienceworld_profile=command_profile(
                            self._config.arm.task_semantic_guidance
                        ),
                    ),
                },
                {"role": "user", "content": task.task},
            ),
            task.task,
            owner_only_system=True,
        )
        context.remember(
            "tool", memory.history[-1].render(), memory.revision, ArchiveKind.ENVIRONMENT
        )
        self._contexts[task.task_id] = context

    async def observe_interactive_episode(
        self, task_id: str, action: str | None, result: NativeEnvironmentStep | NativePublicState
    ) -> None:
        runtime = self._interactive[task_id]
        if isinstance(result, NativeEnvironmentStep):
            observation = result.observation
            status = (
                ExecutionStatus.REJECTED
                if result.action_valid is False
                else ExecutionStatus.CONFIRMED
                if result.action_valid is True
                else ExecutionStatus.ACKNOWLEDGED
            )
            terminal = TaskStatus.TERMINAL if result.terminal else TaskStatus.ACTIVE
            self.counts.tool_calls += 1
        else:
            observation, status, terminal = (
                result.observation_text,
                ExecutionStatus.REJECTED,
                TaskStatus.ACTIVE,
            )
        runtime.memory = runtime.memory.observe(
            action, observation, execution_status=status, task_status=terminal
        )
        runtime.surface = PublicSurface(
            runtime.surface.mode, runtime.memory.revision, result.available_actions
        )
        self.generation.trace(task_id, "public-transition", asdict(runtime.memory.history[-1]))
        self._contexts[task_id].remember(
            "tool",
            runtime.memory.history[-1].render(),
            runtime.memory.revision,
            ArchiveKind.ENVIRONMENT,
        )

    def current_interactive_memory(self, task_id: str) -> str:
        return self._interactive[task_id].memory.render()

    async def finish_interactive_episode(
        self, task_id: str, outcome: NativeEnvironmentOutcome
    ) -> ArchitectureEpisodeArtifact:
        runtime = self._interactive[task_id]
        artifact = ArchitectureEpisodeArtifact(
            f"integrity:{task_id}",
            task_id,
            runtime.turns,
            runtime.turns,
            self._retrieved_ids(),
            outcome.success,
        )
        await self.close_public_episode(task_id, reason="terminal")
        return artifact

    async def close_public_episode(self, task_id: str, *, reason: str) -> None:
        """Close public state without accepting success/reward from the actor."""
        if reason not in {
            "terminal",
            "model-budget",
            "turn-call-budget",
            "action-budget",
            "unsubmitted",
            "infrastructure",
            "owner-finish",
            "wall-clock",
        }:
            raise ValueError("unknown public episode stop reason")
        runtime = self._interactive.pop(task_id, None)
        self.generation.trace(
            task_id,
            "public-episode-close",
            {
                "reason": reason,
                "public_revision": runtime.surface.revision if runtime else 0,
                "attempted_actions": [
                    event.action for event in runtime.memory.history if event.action is not None
                ]
                if runtime
                else [],
            },
        )
        self._contexts.pop(task_id, None)
        self._decisions.pop(task_id, None)
        self._reasoning_requested.discard(task_id)
        self._submissions.pop(task_id, None)
        self._cached = {
            key: value
            for key, value in self._cached.items()
            if self._binding(key)[0].task_id != task_id
        }


def _terminal_mode(binding: StepZeroTaskBinding) -> StepZeroTerminalMode | None:
    return {
        "short-answer": StepZeroTerminalMode.SHORT_ANSWER,
        "aime-boxed-integer": StepZeroTerminalMode.AIME_INTEGER,
        "natural-language": StepZeroTerminalMode.NATURAL_LANGUAGE,
        "python-source": StepZeroTerminalMode.PYTHON_SOURCE,
    }.get(binding.completion_mode.value)
