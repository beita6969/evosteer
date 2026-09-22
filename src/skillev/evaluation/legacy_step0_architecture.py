"""Seeded SkillFlow controller at optimizer step zero.

The bridge keeps benchmark-native generation and scoring outside the
architecture while exercising task-conditioned seed retrieval, a compact H0,
frozen reasoning, and typed terminal/native-action adapters.
Training raw-softmax sampling is intentionally unavailable here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from skillev.contracts import JsonValue, TrajectoryStep
from skillev.evolution import SkillRetrievalDecision, TaskConditionedSkillRetriever
from skillev.policy import encode_rollout_prompt
from skillev.rollout import (
    ACTION_SURFACE_FORMAT_V2,
    ActionSurface,
    ArgumentFieldSpec,
    ArgumentType,
    CanonicalInitialContextAssembler,
    GenerationPhase,
    ModelVisibleMessage,
    RolloutTask,
    RolloutTokenizerProtocol,
    StructuredJsonActionCodec,
    TerminalMode,
    ToolActionSpecV2,
    decode_action_segment,
    decode_reasoning_segment,
)
from skillev.rollout.artifact import (
    ActionDraft,
    CompletedStepDraft,
    ReasoningDraft,
    materialize_trajectory_step,
)
from skillev.rollout.evaluation_sglang import EvaluationRolloutGenerator
from skillev.rollout.prompt_profiles import InitialContextProfile
from skillev.runtime import (
    ActionKind,
    ActionParseResult,
    ActionParseStatus,
    EnvironmentObservation,
    SkillLibrary,
    SkillLibraryState,
    StructuredAction,
)
from skillev.scoring import render_forward_prefix_from_parts, render_step_zero_reasoning_prefix

from .direct_baseline import DirectGenerationError, DirectGenerationRequest, DirectGenerationResult
from .direct_baseline.interactive_tasks import (
    NativeEnvironmentOutcome,
    NativeEnvironmentStep,
    NativeInteractiveTask,
    NativePublicState,
    action_matches_public_surface,
)
from .direct_baseline.parsing import parse_visible_react
from .direct_baseline.prompts import WebShopPublicCatalogCandidate, is_structured_memory_profile
from .direct_baseline.webshop_prompt_policy import is_webshop_thought_free_profile
from .legacy_step0_completion import (
    StepZeroTerminalMode,
    evaluation_request,
    native_action_constraint,
    project_terminal_candidate,
    render_terminal_prefix,
    terminal_constraint,
)
from .legacy_symbolic_policies import (
    _alfworld_go_actions_matching_entity as _alfworld_go_actions_matching_entity,
)
from .legacy_symbolic_policies import (
    _blocked_native_actions as _blocked_native_actions,
)
from .legacy_symbolic_policies import (
    _effective_alfworld_actions as _effective_alfworld_actions,
)
from .legacy_symbolic_policies import (
    _effective_native_actions as _effective_native_actions,
)
from .legacy_symbolic_policies import (
    _public_price_exceeds_limit as _public_price_exceeds_limit,
)
from .legacy_symbolic_policies import (
    _reasoning_aligned_actions as _reasoning_aligned_actions,
)
from .legacy_symbolic_policies import (
    _repeats_webshop_search as _repeats_webshop_search,
)
from .legacy_symbolic_policies import (
    _required_catalog_option_actions as _required_catalog_option_actions,
)
from .legacy_symbolic_policies import (
    _visible_webshop_results as _visible_webshop_results,
)
from .legacy_symbolic_policies import (
    _webshop_catalog_navigation_action as _webshop_catalog_navigation_action,
)
from .legacy_symbolic_policies import (
    _webshop_search_has_price_prose as _webshop_search_has_price_prose,
)
from .step0_conditions import (
    default_step_zero_retrieval_policy,
    matched_step_zero_condition,
)
from .step0_interactive_agent import (
    ArchitectureEpisodeArtifact,
    EpisodeMemoryState,
    StepZeroEpisodeState,
    initial_episode_memory,
    update_episode_memory,
)
from .step0_receipts import ArchitectureIdentityReceipt
from .step0_types import (
    AIME_STEP_ZERO_REASONING_TOKEN_FLOOR,
    ArchitectureInferenceState,
    StepZeroActionMode,
    StepZeroArchitectureConfig,
    StepZeroCompletionMode,
    StepZeroDiagnostics,
    StepZeroReasoningMode,
    StepZeroTaskBinding,
)


def _reasoning_token_budget(
    binding: StepZeroTaskBinding,
    config: StepZeroArchitectureConfig,
) -> int:
    """Resolve the answer-free inference budget for one Step-0 task.

    The official Qwen3.5 AIME condition permits 81,920 reasoning tokens. The
    earlier Step-0 adapter silently capped the same frozen model at 32,768,
    truncating hard solutions before the integer terminal pass. Expanding this
    budget changes neither the task text nor the retrieved skill and exposes
    no evaluator material.
    """

    requested = binding.max_reasoning_tokens or config.max_reasoning_tokens
    if binding.benchmark_id == "aime-2026":
        return max(requested, AIME_STEP_ZERO_REASONING_TOKEN_FLOOR)
    return requested


@dataclass(slots=True)
class _InteractiveRuntime:
    state: StepZeroEpisodeState
    binding: StepZeroTaskBinding
    snapshot_id: str
    retrieved: SkillRetrievalDecision
    prompt_profile_id: str
    parser_profile_id: str
    webshop_catalog_candidates: tuple[WebShopPublicCatalogCandidate, ...] = ()
    current_turn_steps: tuple[TrajectoryStep, ...] = ()
    pending: CompletedStepDraft | None = None
    pending_initial_text: str | None = None


@dataclass(frozen=True, slots=True)
class _TerminalAction:
    text: str
    finish_reason: str


@runtime_checkable
class _ThinkingPromptTokenizer(Protocol):
    def encode_rollout_prompt_with_thinking(self, text: str) -> list[int]: ...

    def encode_step_zero_reasoning_prompt(
        self,
        text: str,
        *,
        enable_thinking: bool,
    ) -> list[int]: ...


@runtime_checkable
class _TerminalPromptTokenizer(Protocol):
    def encode_step_zero_terminal_prompt(self, text: str) -> list[int]: ...


class StepZeroArchitectureDirectClient:
    """Direct runner for the BayesianImprove architecture at optimizer step zero."""

    def __init__(
        self,
        *,
        generator: EvaluationRolloutGenerator,
        library_state: SkillLibraryState,
        bindings: tuple[StepZeroTaskBinding, ...],
        config: StepZeroArchitectureConfig,
    ) -> None:
        if not isinstance(generator, EvaluationRolloutGenerator):
            raise TypeError("Step-0 evaluation requires an explicit evaluation generator")
        if not isinstance(library_state, SkillLibraryState):
            raise TypeError("Step-0 evaluation requires a SkillLibraryState")
        if any(not isinstance(item, StepZeroTaskBinding) for item in bindings):
            raise ValueError("Step-0 evaluation requires typed task bindings")
        if not config.arm.legacy:
            raise ValueError("historical controller requires an explicit legacy arm")
        for binding in bindings:
            if (
                binding.benchmark_id in {"aime-2026", "humaneval", "mbpp-plus"}
                and binding.reasoning_mode is not StepZeroReasoningMode.QWEN_THINKING
            ):
                raise ValueError("historical AIME/code bindings used the thinking template")
        task_ids = tuple(item.task_id for item in bindings)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("Step-0 task bindings must be unique")
        if not isinstance(config, StepZeroArchitectureConfig):
            raise TypeError("Step-0 evaluation requires a typed config")
        self._generator = generator
        self._library = SkillLibrary(library_state)
        self._retriever = TaskConditionedSkillRetriever(library=self._library)
        self._retrieval_policy = default_step_zero_retrieval_policy()
        self._bindings = {item.task_id: item for item in bindings}
        self._config = config
        self._assembler = CanonicalInitialContextAssembler(
            maximum_h0_tokens=config.maximum_h0_tokens
        )
        self._codec = StructuredJsonActionCodec()
        self._interactive: dict[str, _InteractiveRuntime] = {}
        self._static_root_queries = {
            item.task_id: item.root_query for item in bindings if item.root_query is not None
        }
        self._generations = 0
        self._public_catalog_actions = 0
        self._wire_invalid = 0
        self._semantic_invalid = 0
        self._retrieval_abstentions = 0
        self._retrieved_skill_ids: set[str] = set()

    def register_binding(self, binding: StepZeroTaskBinding) -> None:
        """Register one answer-free task route before its first generation.

        Evaluation runners commonly hydrate private benchmark cases lazily.  A
        typed registration boundary lets one persistent architecture client
        serve those cases without rebuilding the client at every native turn.
        Existing bindings are immutable: an identical repeat is harmless, but
        changing a route after registration is rejected.
        """

        if not isinstance(binding, StepZeroTaskBinding):
            raise TypeError("Step-0 task binding must be typed")
        existing = self._bindings.get(binding.task_id)
        if existing is not None and existing != binding:
            raise ValueError("Step-0 task binding cannot change after registration")
        if binding.task_id in self._interactive:
            raise ValueError("Step-0 task binding cannot change during an episode")
        self._bindings[binding.task_id] = binding
        if binding.root_query is not None:
            self._static_root_queries.setdefault(binding.task_id, binding.root_query)

    @property
    def diagnostics(self) -> StepZeroDiagnostics:
        return StepZeroDiagnostics(
            generations=self._generations,
            public_catalog_actions=self._public_catalog_actions,
            wire_invalid=self._wire_invalid,
            semantic_invalid=self._semantic_invalid,
            retrieval_abstentions=self._retrieval_abstentions,
            retrieved_skill_ids=tuple(sorted(self._retrieved_skill_ids)),
        )

    @property
    def architecture_identity(self) -> ArchitectureIdentityReceipt:
        authorities = {item.reasoning_authority_mode for item in self._bindings.values()}
        reasoning_authority = (
            next(iter(authorities)).value
            if len(authorities) == 1
            else "frozen-benchmark-conditioned-mixed"
        )
        state = self._config.inference_state
        if state.forward_adapter_active:
            reasoning_authority = reasoning_authority.replace(
                "frozen-", "trained-forward-policy-", 1
            )
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
            controller_id=self._config.controller_id,
            optimizer_steps=state.optimizer_steps,
            forward_adapter_active=state.forward_adapter_active,
            backward_adapter_active=state.backward_adapter_active,
            posterior_active=state.posterior_active,
            calibration_active=state.calibration_active,
            operator_active=state.operator_active,
            seed_library_id=self._config.seed_library_id,
            seed_skill_ids=self._library.active_skill_ids,
            retrieval_policy_id=self._retrieval_policy.policy_id,
            initial_context_profile=InitialContextProfile.SEEDED_STEP_ZERO.value,
            reasoning_authority=reasoning_authority,
            reasoning_contract_resolved=True,
            action_policy_authority=state.action_policy_authority,
        )

    async def generate(self, request: DirectGenerationRequest) -> DirectGenerationResult:
        binding, outer_step = self._binding(request.request_id)
        try:
            if binding.action_mode in {StepZeroActionMode.WEB_SHOP, StepZeroActionMode.ALF_WORLD}:
                runtime = self._interactive.get(binding.task_id)
                if runtime is None:
                    raise DirectGenerationError("interactive Step-0 episode was not begun")
                return await self._run_interactive(request, outer_step=outer_step, runtime=runtime)
            return await self._run_static(request, binding=binding, outer_step=outer_step)
        except DirectGenerationError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise DirectGenerationError(
                f"Step-0 architecture generation failed: {type(exc).__name__}"
            ) from exc

    async def begin_interactive_episode(
        self,
        task: NativeInteractiveTask,
        state: NativePublicState,
    ) -> None:
        binding = self._bindings.get(task.task_id)
        if binding is None or binding.action_mode not in {
            StepZeroActionMode.WEB_SHOP,
            StepZeroActionMode.ALF_WORLD,
        }:
            raise ValueError("interactive task has no Step-0 binding")
        if task.task_id in self._interactive:
            raise ValueError("interactive Step-0 episode is already active")
        if not is_structured_memory_profile(task.prompt_profile_id):
            raise ValueError("interactive Step-0 requires a structured-memory prompt profile")
        snapshot = self._generator.snapshot()
        root_query = binding.root_query or task.task
        probe_task = _rollout_task(None, binding, outer_step=1, root_query=root_query)
        retrieved = self._retrieve(probe_task, binding)
        episode_id = f"step-zero:{task.task_id}"
        self._generator.begin_episode(episode_id, snapshot.snapshot_id)
        memory = initial_episode_memory(
            benchmark=binding.benchmark_id,
            task_text=task.task,
            task_type=task.task_type,
        )
        self._interactive[task.task_id] = _InteractiveRuntime(
            state=StepZeroEpisodeState(
                episode_id=episode_id,
                task_id=task.task_id,
                root_query=root_query,
                policy_snapshot_id=snapshot.snapshot_id,
                library_version=self._library.current_version,
                retrieved_skill_ids=tuple(item.metadata.skill_id for item in retrieved.selected),
                invoked_skill_ids=set(),
                controller_memory=memory,
                outer_step_index=0,
                architecture_turn_count=0,
                current_observation=state.observation_text,
                current_available_actions=state.available_actions,
            ),
            binding=binding,
            snapshot_id=snapshot.snapshot_id,
            retrieved=retrieved,
            prompt_profile_id=task.prompt_profile_id,
            parser_profile_id=task.parser_profile_id,
            webshop_catalog_candidates=task.webshop_catalog_candidates,
        )

    async def observe_interactive_episode(
        self,
        task_id: str,
        action: str | None,
        result: NativeEnvironmentStep | NativePublicState,
    ) -> None:
        runtime = self._interactive.get(task_id)
        if runtime is None:
            raise ValueError("interactive Step-0 episode is not active")
        if isinstance(result, NativeEnvironmentStep):
            observation = result.observation
            available = result.available_actions
            status = "success" if result.action_valid is not False else "tool_error"
        elif isinstance(result, NativePublicState):
            observation = result.observation_text
            available = result.available_actions
            status = "candidate_invalid"
        else:
            raise TypeError("interactive observation has an incompatible type")
        catalog_options: tuple[tuple[str, tuple[str, ...]], ...] = ()
        webshop_ledger = runtime.state.controller_memory.webshop
        if webshop_ledger is not None and webshop_ledger.selected_asin is not None:
            catalog_options = next(
                (
                    candidate.options
                    for candidate in runtime.webshop_catalog_candidates
                    if candidate.product_id.casefold() == webshop_ledger.selected_asin.casefold()
                ),
                (),
            )
        memory = update_episode_memory(
            runtime.state.controller_memory,
            action=action,
            observation=observation,
            available_actions=available,
            terminal_success=(
                result.success if isinstance(result, NativeEnvironmentStep) else None
            ),
            webshop_option_groups=catalog_options,
        )
        if runtime.pending is not None:
            if runtime.pending_initial_text is None:
                raise RuntimeError("pending interactive step lost its H0")
            materialize_trajectory_step(
                initial_text=runtime.pending_initial_text,
                previous_steps=runtime.current_turn_steps,
                draft=CompletedStepDraft(
                    reasoning=runtime.pending.reasoning,
                    action=runtime.pending.action,
                    observation=EnvironmentObservation(
                        public_value={
                            "action": action,
                            "available_actions": list(available),
                            "controller_memory": memory.render(),
                            "observation": observation,
                        },
                        observation_status=status,
                    ),
                ),
            )
            runtime.pending = None
            runtime.pending_initial_text = None
        runtime.current_turn_steps = ()
        runtime.state.controller_memory = memory
        runtime.state.current_observation = observation
        runtime.state.current_available_actions = available

    def current_interactive_memory(self, task_id: str) -> str:
        """Return controller state after the latest observed environment transition."""

        runtime = self._interactive.get(task_id)
        if runtime is None:
            raise ValueError("interactive Step-0 episode is not active")
        return runtime.state.controller_memory.render()

    async def finish_interactive_episode(
        self,
        task_id: str,
        outcome: NativeEnvironmentOutcome,
    ) -> ArchitectureEpisodeArtifact:
        runtime = self._interactive.pop(task_id, None)
        if runtime is None:
            raise ValueError("interactive Step-0 episode is not active")
        try:
            if runtime.pending is not None:
                raise RuntimeError("interactive Step-0 episode has an unobserved action")
            return ArchitectureEpisodeArtifact(
                episode_id=runtime.state.episode_id,
                task_id=task_id,
                outer_steps=runtime.state.outer_step_index,
                architecture_turns=runtime.state.architecture_turn_count,
                retrieved_skill_ids=runtime.state.retrieved_skill_ids,
                terminal_success=outcome.success,
            )
        finally:
            self._generator.end_episode(runtime.state.episode_id)

    async def _run_static(
        self,
        request: DirectGenerationRequest,
        *,
        binding: StepZeroTaskBinding,
        outer_step: int,
    ) -> DirectGenerationResult:
        derived_root = binding.root_query or _root_query_from_messages(
            request.messages, benchmark_id=binding.benchmark_id
        )
        root_query = self._static_root_queries.setdefault(binding.task_id, derived_root)
        task = _rollout_task(request, binding, outer_step=outer_step, root_query=root_query)
        retrieved = self._retrieve(task, binding)
        assembled = self._assemble(task, retrieved, binding)
        snapshot = self._generator.snapshot()
        episode_id = f"step-zero:{request.request_id}"
        self._generator.begin_episode(episode_id, snapshot.snapshot_id)
        try:
            if (
                binding.action_mode is StepZeroActionMode.TRIVIA_SEARCH
                and outer_step <= binding.maximum_trivia_searches
            ):
                return await self._run_native_action(
                    request,
                    binding=binding,
                    outer_step=outer_step,
                    assembled=assembled,
                    snapshot_id=snapshot.snapshot_id,
                    retrieved=retrieved,
                    completed_steps=(),
                    memory=None,
                    current_actions=(),
                    prompt_profile_id=None,
                    persist_runtime=None,
                )
            return await self._run_terminal(
                request,
                binding=binding,
                outer_step=outer_step,
                assembled=assembled,
                snapshot_id=snapshot.snapshot_id,
            )
        finally:
            self._generator.end_episode(episode_id)

    async def _run_interactive(
        self,
        request: DirectGenerationRequest,
        *,
        outer_step: int,
        runtime: _InteractiveRuntime,
    ) -> DirectGenerationResult:
        if outer_step != runtime.state.outer_step_index + 1:
            raise DirectGenerationError("interactive Step-0 outer step is not contiguous")
        if runtime.pending is not None:
            raise DirectGenerationError("interactive Step-0 action was not observed")
        retrieved_skill_ids = set(runtime.state.retrieved_skill_ids)
        webshop_skill_active = "skill-webshop-constraint-ledger" in retrieved_skill_ids
        alfworld_skill_active = "skill-alfworld-subgoal-machine" in retrieved_skill_ids
        skill_orchestration_active = webshop_skill_active or alfworld_skill_active
        current_actions = (
            _effective_native_actions(
                runtime.binding,
                runtime.state.controller_memory,
                runtime.state.current_available_actions,
                runtime.state.current_observation,
            )
            if skill_orchestration_active
            else runtime.state.current_available_actions
        )
        if webshop_skill_active:
            catalog_action = _webshop_catalog_navigation_action(
                runtime.webshop_catalog_candidates,
                runtime.state.controller_memory,
                current_actions,
                runtime.state.current_observation,
                option_surface=runtime.state.current_available_actions,
            )
            if catalog_action is not None:
                runtime.state.outer_step_index = outer_step
                runtime.state.invoked_skill_ids.add("skill-webshop-constraint-ledger")
                self._public_catalog_actions += 1
                thought = (
                    "The retrieved WebShop skill's public catalog operator selected this "
                    "currently executable action from public task, product, and live-surface "
                    "evidence."
                )
                return DirectGenerationResult(
                    request_id=request.request_id,
                    text=_render_visible_react(
                        memory=runtime.state.controller_memory.render(),
                        reasoning=thought,
                        action=catalog_action,
                        include_thought=not is_webshop_thought_free_profile(
                            runtime.prompt_profile_id
                        ),
                    ),
                    finish_reason="step-zero-retrieved-skill-operator-action",
                    prompt_tokens=0,
                    completion_tokens=0,
                    reasoning_text=thought,
                    response_model="skill-webshop-constraint-ledger/operator@1",
                    response_id=f"{request.request_id}:retrieved-skill-operator-action",
                    service_instance_id=self._config.service_instance_id,
                )
        task = _rollout_task(
            request,
            runtime.binding,
            outer_step=outer_step,
            root_query=runtime.state.root_query,
            current_actions=current_actions,
            controller_memory=runtime.state.controller_memory,
            webshop_catalog_candidates=runtime.webshop_catalog_candidates,
        )
        assembled = self._assemble(task, runtime.retrieved, runtime.binding)
        runtime.state.outer_step_index = outer_step
        return await self._run_native_action(
            request,
            binding=runtime.binding,
            outer_step=outer_step,
            assembled=assembled,
            snapshot_id=runtime.snapshot_id,
            retrieved=runtime.retrieved,
            completed_steps=runtime.current_turn_steps,
            memory=runtime.state.controller_memory,
            current_actions=current_actions,
            prompt_profile_id=runtime.prompt_profile_id,
            persist_runtime=runtime,
        )

    async def _run_terminal(
        self,
        request: DirectGenerationRequest,
        *,
        binding: StepZeroTaskBinding,
        outer_step: int,
        assembled: object,
        snapshot_id: str,
    ) -> DirectGenerationResult:
        initial_text = _assembled_text(assembled)
        reasoning_text, _, prompt_tokens, completion_tokens = await self._reason(
            request,
            binding=binding,
            outer_step=outer_step,
            internal_step=1,
            initial_text=initial_text,
            completed_steps=(),
            snapshot_id=snapshot_id,
        )
        terminal_mode = _terminal_mode(binding)
        terminal_prompt = render_terminal_prefix(
            initial_text,
            reasoning_text=reasoning_text,
            mode=terminal_mode,
            public_question=(
                binding.root_query
                or _root_query_from_messages(
                    request.messages,
                    benchmark_id=binding.benchmark_id,
                )
            ),
        )
        action_tokens = binding.max_action_tokens or min(
            request.profile.max_new_tokens,
            self._config.max_action_tokens,
        )
        condition = matched_step_zero_condition(
            request.profile,
            condition_id=f"{binding.benchmark_id}:step0@3",
            reasoning_tokens=_reasoning_token_budget(binding, self._config),
            action_tokens=action_tokens,
            reasoning_authority_mode=binding.reasoning_authority_mode,
            deterministic_action=(
                terminal_mode
                in {StepZeroTerminalMode.AIME_INTEGER, StepZeroTerminalMode.SHORT_ANSWER}
            ),
        )
        action_input_ids = tuple(
            _encode_terminal_prompt(self._generator.tokenizer, terminal_prompt)
        )
        action_result = await self._generator.generate_evaluation(
            evaluation_request(
                input_ids=action_input_ids,
                max_new_tokens=condition.action.max_new_tokens,
                seed=condition.action.seed,
                decoding_profile_id=condition.action.profile_id,
                policy_snapshot_id=snapshot_id,
                phase=GenerationPhase.ACTION,
            ),
            profile=condition.action,
            constraint=terminal_constraint(terminal_mode),
        )
        self._generations += 1
        prompt_tokens += action_result.usage.input_tokens
        completion_tokens += action_result.usage.output_tokens
        candidate_ids = action_result.content_token_ids or action_result.stop_token_ids
        if not candidate_ids:
            self._wire_invalid += 1
            return _candidate_failure(
                request,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                reasoning_text=reasoning_text,
                config=self._config,
                reason="empty-terminal",
            )
        candidate = decode_action_segment(self._generator.tokenizer, candidate_ids).text
        projected = project_terminal_candidate(terminal_mode, candidate)
        if projected is None:
            self._wire_invalid += 1
            return _candidate_failure(
                request,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                reasoning_text=reasoning_text,
                config=self._config,
                reason="invalid-terminal",
            )
        finish = (
            "step-zero-trivia-final"
            if binding.action_mode is StepZeroActionMode.TRIVIA_SEARCH
            else "step-zero-typed-terminal"
        )
        return _direct_result(
            request=request,
            terminal=_TerminalAction(projected, finish),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_texts=(reasoning_text,),
            config=self._config,
        )

    async def _run_native_action(
        self,
        request: DirectGenerationRequest,
        *,
        binding: StepZeroTaskBinding,
        outer_step: int,
        assembled: object,
        snapshot_id: str,
        retrieved: SkillRetrievalDecision,
        completed_steps: tuple[TrajectoryStep, ...],
        memory: EpisodeMemoryState | None,
        current_actions: tuple[str, ...],
        prompt_profile_id: str | None,
        persist_runtime: _InteractiveRuntime | None,
    ) -> DirectGenerationResult:
        initial_text = _assembled_text(assembled)
        working_steps = completed_steps
        prompt_tokens = completion_tokens = 0
        reasoning_texts: list[str] = []
        for _ in range(self._config.max_turns):
            internal_step = len(working_steps) + 1
            reasoning_text, reasoning_draft, used_prompt, used_completion = await self._reason(
                request,
                binding=binding,
                outer_step=outer_step,
                internal_step=internal_step,
                initial_text=initial_text,
                completed_steps=working_steps,
                snapshot_id=snapshot_id,
            )
            prompt_tokens += used_prompt
            completion_tokens += used_completion
            reasoning_texts.append(reasoning_text)
            forward = render_forward_prefix_from_parts(
                initial_text,
                working_steps,
                internal_step,
                reasoning_text,
            )
            action_tokens = binding.max_action_tokens or min(
                request.profile.max_new_tokens,
                self._config.max_action_tokens,
            )
            condition = matched_step_zero_condition(
                request.profile,
                condition_id=f"{binding.benchmark_id}:step0@3",
                reasoning_tokens=_reasoning_token_budget(binding, self._config),
                action_tokens=action_tokens,
                reasoning_authority_mode=binding.reasoning_authority_mode,
            )
            action_input_ids = tuple(encode_rollout_prompt(self._generator.tokenizer, forward.text))
            constraint_mode = (
                "trivia-search"
                if binding.action_mode is StepZeroActionMode.TRIVIA_SEARCH
                else binding.action_mode.value
            )
            action_actions = _reasoning_aligned_actions(
                binding,
                current_actions,
                reasoning_text,
                memory=memory,
            )
            action_result = await self._generator.generate_evaluation(
                evaluation_request(
                    input_ids=action_input_ids,
                    max_new_tokens=condition.action.max_new_tokens,
                    seed=condition.action.seed,
                    decoding_profile_id=condition.action.profile_id,
                    policy_snapshot_id=snapshot_id,
                    phase=GenerationPhase.ACTION,
                ),
                profile=condition.action,
                constraint=native_action_constraint(
                    constraint_mode,
                    available_actions=action_actions,
                    blocked_actions=(),
                ),
            )
            self._generations += 1
            prompt_tokens += action_result.usage.input_tokens
            completion_tokens += action_result.usage.output_tokens
            action_ids = action_result.content_token_ids or action_result.stop_token_ids
            if not action_ids:
                self._wire_invalid += 1
                if persist_runtime is not None:
                    added_internal_turns = len(working_steps) - len(
                        persist_runtime.current_turn_steps
                    )
                    persist_runtime.current_turn_steps = working_steps
                    persist_runtime.state.architecture_turn_count += added_internal_turns + 1
                return _candidate_failure(
                    request,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    reasoning_text="\n\n".join(reasoning_texts),
                    config=self._config,
                    reason="empty-action",
                )
            segment = decode_action_segment(self._generator.tokenizer, action_ids)
            parsed = self._codec.parse(segment.text)
            native = _native_action(binding, outer_step, parsed.status, parsed.action)
            if native is None:
                if parsed.status is not ActionParseStatus.VALID:
                    self._wire_invalid += 1
                    code = "action_wire_invalid"
                else:
                    self._semantic_invalid += 1
                    code = "action_semantic_invalid"
                working_steps = _commit_internal_step(
                    initial_text,
                    working_steps,
                    reasoning_draft,
                    forward.text,
                    forward.prefix_hash,
                    segment.text,
                    segment.token_ids,
                    parsed,
                    snapshot_id,
                    _invalid_observation("schema_invalid", code),
                )
                continue
            if action_actions and not _action_on_surface(binding, native, action_actions):
                self._semantic_invalid += 1
                working_steps = _commit_internal_step(
                    initial_text,
                    working_steps,
                    reasoning_draft,
                    forward.text,
                    forward.prefix_hash,
                    segment.text,
                    segment.token_ids,
                    parsed,
                    snapshot_id,
                    _invalid_observation("tool_error", "action_not_on_current_surface"),
                )
                continue
            draft = CompletedStepDraft(
                reasoning=reasoning_draft,
                action=ActionDraft(
                    step_index=internal_step,
                    forward_prefix_text=forward.text,
                    forward_prefix_hash=forward.prefix_hash,
                    text=segment.text,
                    token_ids=segment.token_ids,
                    parse_result=parsed,
                    policy_snapshot_id=snapshot_id,
                ),
                observation=_invalid_observation("other", "awaiting_native_environment"),
            )
            if persist_runtime is not None:
                added_internal_turns = len(working_steps) - len(persist_runtime.current_turn_steps)
                persist_runtime.current_turn_steps = working_steps
                persist_runtime.pending = draft
                persist_runtime.pending_initial_text = initial_text
                persist_runtime.state.architecture_turn_count += added_internal_turns + 1
            if binding.action_mode is StepZeroActionMode.TRIVIA_SEARCH:
                text = f"Search: {native.removeprefix('search[').removesuffix(']')}"
                finish = "step-zero-trivia-search"
            else:
                fallback_memory = (
                    memory.render() if memory is not None else "Public state retained."
                )
                rendered_memory, visible_thought = _project_visible_react_state(
                    reasoning_text,
                    fallback_memory=fallback_memory,
                )
                text = _render_visible_react(
                    memory=rendered_memory,
                    reasoning=visible_thought,
                    action=native,
                    include_thought=(
                        prompt_profile_id is None
                        or not is_webshop_thought_free_profile(prompt_profile_id)
                    ),
                )
                finish = "step-zero-native-action"
            return _direct_result(
                request=request,
                terminal=_TerminalAction(text, finish),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                reasoning_texts=tuple(reasoning_texts),
                config=self._config,
            )
        if persist_runtime is not None:
            added_internal_turns = len(working_steps) - len(persist_runtime.current_turn_steps)
            persist_runtime.current_turn_steps = working_steps
            persist_runtime.state.architecture_turn_count += added_internal_turns
        return _candidate_failure(
            request,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_text="\n\n".join(reasoning_texts),
            config=self._config,
            reason="architecture-horizon",
        )

    async def _reason(
        self,
        request: DirectGenerationRequest,
        *,
        binding: StepZeroTaskBinding,
        outer_step: int,
        internal_step: int,
        initial_text: str,
        completed_steps: tuple[TrajectoryStep, ...],
        snapshot_id: str,
    ) -> tuple[str, ReasoningDraft, int, int]:
        del outer_step
        prompt = render_step_zero_reasoning_prefix(initial_text, completed_steps, internal_step)
        input_ids = tuple(
            _encode_reasoning_prompt(
                self._generator.tokenizer,
                prompt.text,
                mode=binding.reasoning_mode,
            )
        )
        token_budget = _reasoning_token_budget(binding, self._config)
        condition = matched_step_zero_condition(
            request.profile,
            condition_id=f"{binding.benchmark_id}:step0@3",
            reasoning_tokens=token_budget,
            action_tokens=(binding.max_action_tokens or self._config.max_action_tokens),
            reasoning_authority_mode=binding.reasoning_authority_mode,
        )
        result = await self._generator.generate_evaluation(
            evaluation_request(
                input_ids=input_ids,
                max_new_tokens=condition.reasoning.max_new_tokens,
                seed=condition.reasoning.seed,
                decoding_profile_id=condition.reasoning.profile_id,
                policy_snapshot_id=snapshot_id,
                phase=GenerationPhase.REASONING,
            ),
            profile=condition.reasoning,
        )
        self._generations += 1
        decoded = decode_reasoning_segment(self._generator.tokenizer, result.content_token_ids)
        return (
            decoded.text,
            ReasoningDraft(
                step_index=internal_step,
                prompt_text=prompt.text,
                prompt_hash=prompt.prompt_hash,
                text=decoded.text,
                generated_token_ids=decoded.token_ids,
                policy_snapshot_id=snapshot_id,
            ),
            result.usage.input_tokens,
            result.usage.output_tokens,
        )

    def _retrieve(
        self,
        task: RolloutTask,
        binding: StepZeroTaskBinding,
    ) -> SkillRetrievalDecision:
        decision = self._retriever.retrieve_step_zero(
            task,
            policy=self._retrieval_policy,
            tokenizer=self._generator.tokenizer,
            instruction_token_budget=binding.skill_instruction_token_budget,
        )
        if decision.abstained:
            self._retrieval_abstentions += 1
        self._retrieved_skill_ids.update(item.metadata.skill_id for item in decision.selected)
        return decision

    def _assemble(
        self,
        task: RolloutTask,
        retrieved: SkillRetrievalDecision,
        binding: StepZeroTaskBinding,
    ) -> object:
        assembler = (
            self._assembler
            if binding.maximum_h0_tokens is None
            else CanonicalInitialContextAssembler(maximum_h0_tokens=binding.maximum_h0_tokens)
        )
        return assembler.assemble(
            task=task,
            retrieved_skills=retrieved.selected,
            active_skill_ids=self._library.active_skill_ids,
            library_version=self._library.current_version,
            tokenizer=self._generator.tokenizer,
            profile=InitialContextProfile.SEEDED_STEP_ZERO,
        )

    def _binding(self, request_id: str) -> tuple[StepZeroTaskBinding, int]:
        exact = self._bindings.get(request_id)
        if exact is not None:
            return exact, 0
        matches: list[tuple[int, StepZeroTaskBinding, str]] = []
        for task_id, binding in self._bindings.items():
            for marker in (":step:", ":search-turn:"):
                prefix = task_id + marker
                if request_id.startswith(prefix):
                    matches.append((len(task_id), binding, request_id[len(prefix) :]))
        if not matches:
            raise DirectGenerationError("Step-0 request has no task binding")
        longest = max(item[0] for item in matches)
        selected = tuple(item for item in matches if item[0] == longest)
        if len(selected) != 1:
            raise DirectGenerationError("Step-0 request binding is ambiguous")
        _, binding, suffix = selected[0]
        if not suffix.isdecimal() or int(suffix) < 1:
            raise DirectGenerationError("Step-0 outer turn identity is invalid")
        return binding, int(suffix)


def _assembled_text(value: object) -> str:
    text = getattr(value, "text", None)
    if type(text) is not str or not text:
        raise TypeError("Step-0 assembled context is invalid")
    return text


def _rollout_task(
    request: DirectGenerationRequest | None,
    binding: StepZeroTaskBinding,
    *,
    outer_step: int,
    root_query: str,
    current_actions: tuple[str, ...] = (),
    controller_memory: EpisodeMemoryState | None = None,
    webshop_catalog_candidates: tuple[WebShopPublicCatalogCandidate, ...] = (),
) -> RolloutTask:
    terminal_wire = None
    if binding.action_mode is StepZeroActionMode.COMPLETION or (
        binding.action_mode is StepZeroActionMode.TRIVIA_SEARCH
        and outer_step > binding.maximum_trivia_searches
    ):
        terminal_wire = _terminal_mode(binding).value
    context: dict[str, JsonValue] = {
        "benchmark_id": binding.benchmark_id,
        "condition": "optimizer-step-zero-no-update",
        "reasoning_authority": "frozen-deterministic",
        "skill_exposure_mode": "full-inline-no-invoke",
    }
    if binding.action_mode is StepZeroActionMode.TRIVIA_SEARCH:
        context["retrieval_searches_completed"] = min(
            outer_step - 1,
            binding.maximum_trivia_searches,
        )
        context["retrieval_search_budget"] = binding.maximum_trivia_searches
    if binding.action_mode in {StepZeroActionMode.WEB_SHOP, StepZeroActionMode.ALF_WORLD}:
        context["environment_completion_status"] = "not-terminal-active-decision"
        context["current_native_action_surface"] = list(current_actions)
        if controller_memory is not None:
            context["current_controller_memory"] = controller_memory.render()
            context["controller_memory_authority"] = (
                "current environment-derived state; overrides older source-message memory"
            )
    if binding.action_mode is StepZeroActionMode.WEB_SHOP and webshop_catalog_candidates:
        context["retrieved_public_catalog_candidates"] = [
            _public_catalog_candidate_value(candidate) for candidate in webshop_catalog_candidates
        ]
    if terminal_wire is not None:
        context["step_zero_terminal_wire"] = terminal_wire
    return RolloutTask(
        task_id=binding.task_id,
        environment_id=f"protocol13:{binding.benchmark_id}:step-zero",
        task_family=binding.task_family,
        context_id=f"protocol13:{binding.benchmark_id}:iid",
        query=root_query,
        available_tools=_available_tools(
            binding.action_mode,
            outer_step=outer_step,
            maximum_trivia_searches=binding.maximum_trivia_searches,
            current_actions=current_actions,
        ),
        public_context=context,
        action_surface=_action_surface(
            binding.action_mode,
            outer_step=outer_step,
            maximum_trivia_searches=binding.maximum_trivia_searches,
            current_actions=current_actions,
        ),
        model_visible_messages=(
            ()
            if request is None
            else tuple(
                ModelVisibleMessage(role=item["role"], content=item["content"])
                for item in request.messages
            )
        ),
    )


def _public_catalog_candidate_value(
    candidate: WebShopPublicCatalogCandidate,
) -> dict[str, JsonValue]:
    """Expose answer-independent catalog evidence without turning it into an action."""

    return {
        "attributes": list(candidate.attributes),
        "options": {group: list(values) for group, values in candidate.options},
        "price": candidate.price,
        "product_id": candidate.product_id,
        "suggested_options": dict(candidate.suggested_options),
        "suggested_search_query": candidate.suggested_search_query,
        "title": candidate.title,
    }


def _root_query_from_messages(
    messages: tuple[dict[str, str], ...],
    *,
    benchmark_id: str,
) -> str:
    for message in reversed(messages):
        if message["role"] == "user":
            content = message["content"]
            break
    else:
        content = messages[-1]["content"]
    if benchmark_id == "hotpotqa":
        _context, marker, question = content.rpartition("\n\nQuestion:\n")
        if not marker or not question.strip():
            raise DirectGenerationError(
                "HotpotQA source messages lack the frozen question boundary"
            )
        return question.strip()
    return content


def _available_tools(
    mode: StepZeroActionMode,
    *,
    outer_step: int,
    maximum_trivia_searches: int = 1,
    current_actions: tuple[str, ...] = (),
) -> tuple[str, ...]:
    if mode is StepZeroActionMode.TRIVIA_SEARCH and outer_step <= maximum_trivia_searches:
        return ("search",)
    if mode is StepZeroActionMode.WEB_SHOP:
        tools = []
        if "search" in current_actions or not current_actions:
            tools.append("search")
        if any(action.startswith("click[") for action in current_actions) or not current_actions:
            tools.append("click")
        return tuple(sorted(tools))
    if mode is StepZeroActionMode.ALF_WORLD:
        return ("act",)
    return ()


def _string_argument() -> ArgumentFieldSpec:
    return ArgumentFieldSpec(value_type=ArgumentType.STRING, required=True)


def _action_surface(
    mode: StepZeroActionMode,
    *,
    outer_step: int,
    maximum_trivia_searches: int = 1,
    current_actions: tuple[str, ...] = (),
) -> ActionSurface | None:
    if mode is StepZeroActionMode.TRIVIA_SEARCH and outer_step <= maximum_trivia_searches:
        return ActionSurface(
            terminal_mode=TerminalMode.ENVIRONMENT,
            tools=(
                ToolActionSpecV2(
                    resource_id="triviaqa",
                    name="search",
                    arguments={"query": _string_argument()},
                    example_arguments={"query": "CONCISE_RESEARCH_QUERY"},
                ),
            ),
            public_instructions=(
                "The current public retrieval surface accepts one task-local search query.",
            ),
            format=ACTION_SURFACE_FORMAT_V2,
        )
    if mode is StepZeroActionMode.WEB_SHOP:
        tools: list[ToolActionSpecV2] = []
        if "search" in current_actions or not current_actions:
            tools.append(
                ToolActionSpecV2(
                    resource_id="webshop",
                    name="search",
                    arguments={"query": _string_argument()},
                    example_arguments={"query": "SEARCH_QUERY"},
                )
            )
        click_target = next(
            (
                action[len("click[") : -1]
                for action in current_actions
                if action.startswith("click[") and action.endswith("]")
            ),
            "CURRENT_VISIBLE_TARGET",
        )
        if click_target != "CURRENT_VISIBLE_TARGET" or not current_actions:
            tools.append(
                ToolActionSpecV2(
                    resource_id="webshop",
                    name="click",
                    arguments={"target": _string_argument()},
                    example_arguments={"target": click_target},
                )
            )
        return ActionSurface(
            terminal_mode=TerminalMode.ENVIRONMENT,
            tools=tuple(tools),
            public_instructions=(
                "Choose one exact action from the latest public WebShop action surface.",
            ),
            format=ACTION_SURFACE_FORMAT_V2,
        )
    if mode is StepZeroActionMode.ALF_WORLD:
        return ActionSurface(
            terminal_mode=TerminalMode.ENVIRONMENT,
            tools=(
                ToolActionSpecV2(
                    resource_id="alfworld",
                    name="act",
                    arguments={"command": _string_argument()},
                    example_arguments={"command": "ONE_CURRENTLY_ADMISSIBLE_COMMAND"},
                ),
            ),
            public_instructions=("Copy one exact current admissible command.",),
            format=ACTION_SURFACE_FORMAT_V2,
        )
    return None


def _terminal_mode(binding: StepZeroTaskBinding) -> StepZeroTerminalMode:
    mapping = {
        StepZeroCompletionMode.SHORT_ANSWER: StepZeroTerminalMode.SHORT_ANSWER,
        StepZeroCompletionMode.AIME_BOXED_INTEGER: StepZeroTerminalMode.AIME_INTEGER,
        StepZeroCompletionMode.NATURAL_LANGUAGE: StepZeroTerminalMode.NATURAL_LANGUAGE,
        StepZeroCompletionMode.PYTHON_SOURCE: StepZeroTerminalMode.PYTHON_SOURCE,
    }
    explicit = mapping.get(binding.completion_mode)
    if explicit is not None:
        return explicit
    if binding.benchmark_id in {"hotpotqa", "triviaqa"}:
        return StepZeroTerminalMode.SHORT_ANSWER
    if binding.benchmark_id == "aime-2026":
        return StepZeroTerminalMode.AIME_INTEGER
    if binding.benchmark_id == "healthbench":
        return StepZeroTerminalMode.NATURAL_LANGUAGE
    if binding.benchmark_id in {"mbpp-plus", "humaneval"}:
        return StepZeroTerminalMode.PYTHON_SOURCE
    return StepZeroTerminalMode.NATURAL_LANGUAGE


def _native_action(
    binding: StepZeroTaskBinding,
    outer_step: int,
    status: ActionParseStatus,
    action: StructuredAction | None,
) -> str | None:
    if (
        status is not ActionParseStatus.VALID
        or action is None
        or action.kind is not ActionKind.TOOL
    ):
        return None
    if (
        binding.action_mode is StepZeroActionMode.TRIVIA_SEARCH
        and outer_step <= binding.maximum_trivia_searches
    ):
        if (
            action.resource_id != "triviaqa"
            or action.name != "search"
            or not isinstance(action.arguments, dict)
            or set(action.arguments) != {"query"}
        ):
            return None
        query = action.arguments["query"]
        if type(query) is not str:
            return None
        normalized = " ".join(query.split())
        return f"search[{normalized}]" if normalized and len(normalized) <= 256 else None
    if binding.action_mode is StepZeroActionMode.WEB_SHOP:
        return _webshop_action(action)
    if binding.action_mode is StepZeroActionMode.ALF_WORLD:
        return _alfworld_action(action)
    return None


def _webshop_action(action: StructuredAction) -> str | None:
    if action.resource_id != "webshop" or not isinstance(action.arguments, dict):
        return None
    field = {"search": "query", "click": "target"}.get(action.name)
    if field is None or set(action.arguments) != {field}:
        return None
    value = action.arguments[field]
    if type(value) is not str:
        return None
    normalized = value.strip()
    wrapper = f"{action.name}["
    if normalized.startswith(wrapper) and normalized.endswith("]"):
        normalized = normalized[len(wrapper) : -1].strip()
    if not normalized or "[" in normalized or "]" in normalized:
        return None
    return f"{action.name}[{normalized}]"


def _alfworld_action(action: StructuredAction) -> str | None:
    if (
        action.resource_id != "alfworld"
        or action.name != "act"
        or not isinstance(action.arguments, dict)
        or set(action.arguments) != {"command"}
    ):
        return None
    command = action.arguments["command"]
    return command.strip() if type(command) is str and command.strip() else None


def _action_on_surface(
    binding: StepZeroTaskBinding,
    action: str,
    available_actions: tuple[str, ...],
) -> bool:
    if binding.action_mode is StepZeroActionMode.WEB_SHOP:
        from .direct_baseline.config import DirectBenchmark

        return action_matches_public_surface(DirectBenchmark.WEB_SHOP, action, available_actions)
    return action in available_actions


def _commit_internal_step(
    initial_text: str,
    previous: tuple[TrajectoryStep, ...],
    reasoning: ReasoningDraft,
    forward_text: str,
    forward_hash: str,
    action_text: str,
    action_ids: tuple[int, ...],
    parse_result: ActionParseResult,
    snapshot_id: str,
    observation: EnvironmentObservation,
) -> tuple[TrajectoryStep, ...]:
    return (
        *previous,
        materialize_trajectory_step(
            initial_text=initial_text,
            previous_steps=previous,
            draft=CompletedStepDraft(
                reasoning=reasoning,
                action=ActionDraft(
                    step_index=len(previous) + 1,
                    forward_prefix_text=forward_text,
                    forward_prefix_hash=forward_hash,
                    text=action_text,
                    token_ids=action_ids,
                    parse_result=parse_result,
                    policy_snapshot_id=snapshot_id,
                ),
                observation=observation,
            ),
        ),
    )


def _invalid_observation(status: str, error_code: str) -> EnvironmentObservation:
    return EnvironmentObservation(
        public_value={"error": error_code},
        observation_status=status,
    )


_REACT_SECTION_HEADER = re.compile(r"(?im)^\s*(?:memory|thought|action)\s*:")


def _project_visible_react_state(
    reasoning: str,
    *,
    fallback_memory: str,
) -> tuple[str, str]:
    """Project a benchmark-native memory/thought without trusting its action.

    The source prompt for the current WebShop/ALFWorld turn asks Qwen to update
    structured memory.  Even during the architecture's reasoning pass, Qwen
    often answers that source contract and emits ``Memory/Thought/Action``.
    Previously the entire response was nested inside the outer Thought block,
    creating duplicate section headers; the public parser then discarded the
    memory and sometimes the action.  Preserve only its public-state memory and
    thought.  The separately constrained architecture action remains the sole
    executable wire.
    """

    projected = parse_visible_react(reasoning)
    model_memory = (
        ""
        if "Current product:" in fallback_memory
        else _non_authoritative_working_memory(projected.memory)
    )
    memory = fallback_memory
    if model_memory:
        memory = (
            f"{fallback_memory}\n"
            "Model working memory (cannot override the authoritative constraints above):\n"
            f"{model_memory}"
        )
    thought = projected.thought or reasoning
    return memory, thought


_MODEL_AUTHORITY_FIELD = re.compile(
    r"(?ix)^\s*(?:authoritative\s+)?(?:current\s+)?(?:"
    r"goal(?:\s*/\s*hard\s+constraints?)?"
    r"|task|objective|constraints?|hard\s+constraints?"
    r"|target(?:\s+(?:object|receptacle|destination))?"
    r"|destination|required\s+object"
    r"|remaining\s+subgoals?|completed\s+subgoals?"
    r")\s*:"
)
_MODEL_MEMORY_FIELD = re.compile(r"^\s*[A-Za-z][^:\n]{0,80}:\s*")


def _non_authoritative_working_memory(memory: str | None) -> str:
    """Keep useful progress facts while preventing a demonstration goal override.

    Step-0 controller memory is derived directly from the authoritative current
    task and the native environment.  Model-authored memory is useful for
    visited locations and candidate ledgers, but its Goal/Task/Constraints
    fields are not an authority boundary: a retrieved demonstration can replace
    the current target with its example target. Drop model-owned goal, target,
    destination, and completion-plan fields, and retain only observational
    working state beneath the controller memory.
    """

    if memory is None:
        return ""
    retained: list[str] = []
    dropping_authority_value = False
    for line in memory.splitlines():
        if _MODEL_AUTHORITY_FIELD.match(line):
            dropping_authority_value = True
            continue
        if dropping_authority_value:
            if not _MODEL_MEMORY_FIELD.match(line):
                continue
            dropping_authority_value = False
        retained.append(line.rstrip())
    return "\n".join(retained).strip()


def _render_visible_react(
    *,
    memory: str,
    reasoning: str,
    action: str,
    include_thought: bool = True,
) -> str:
    normalized_memory = _REACT_SECTION_HEADER.sub("", memory).strip()[:4000]
    compact_thought = " ".join(_REACT_SECTION_HEADER.sub("", reasoning).split()).strip()[:1200]
    thought = compact_thought or "Select the current legal progress action."
    if not include_thought:
        return f"Memory:\n{normalized_memory}\n\nAction:\n{action}"
    return f"Memory:\n{normalized_memory}\n\nThought:\n{thought}\n\nAction:\n{action}"


def _encode_reasoning_prompt(
    tokenizer: RolloutTokenizerProtocol,
    text: str,
    *,
    mode: StepZeroReasoningMode,
) -> list[int]:
    if not isinstance(tokenizer, _ThinkingPromptTokenizer):
        raise DirectGenerationError("Step-0 reasoning requires a compatible Qwen tokenizer")
    encoded = tokenizer.encode_step_zero_reasoning_prompt(
        text,
        enable_thinking=mode is StepZeroReasoningMode.QWEN_THINKING,
    )
    if not encoded or any(type(token_id) is not int or token_id < 0 for token_id in encoded):
        raise DirectGenerationError("thinking prompt encoding is invalid")
    return encoded


def _encode_terminal_prompt(
    tokenizer: RolloutTokenizerProtocol,
    text: str,
) -> list[int]:
    if not isinstance(tokenizer, _TerminalPromptTokenizer):
        raise DirectGenerationError("typed Step-0 terminals require a terminal-aware tokenizer")
    encoded = tokenizer.encode_step_zero_terminal_prompt(text)
    if not encoded or any(type(token_id) is not int or token_id < 0 for token_id in encoded):
        raise DirectGenerationError("terminal prompt encoding is invalid")
    return encoded


def _direct_result(
    *,
    request: DirectGenerationRequest,
    terminal: _TerminalAction,
    prompt_tokens: int,
    completion_tokens: int,
    reasoning_texts: tuple[str, ...],
    config: StepZeroArchitectureConfig,
) -> DirectGenerationResult:
    return DirectGenerationResult(
        request_id=request.request_id,
        text=terminal.text,
        finish_reason=terminal.finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_text="\n\n".join(reasoning_texts) or None,
        response_model=config.response_model,
        response_id=f"{request.request_id}:step-zero",
        service_instance_id=config.service_instance_id,
    )


def _candidate_failure(
    request: DirectGenerationRequest,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    reasoning_text: str,
    config: StepZeroArchitectureConfig,
    reason: str,
) -> DirectGenerationResult:
    return DirectGenerationResult(
        request_id=request.request_id,
        text="",
        finish_reason=f"step-zero-candidate-{reason}",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_text=reasoning_text or None,
        response_model=config.response_model,
        response_id=f"{request.request_id}:step-zero-candidate-failure",
        service_instance_id=config.service_instance_id,
    )


__all__ = [
    "AIME_STEP_ZERO_REASONING_TOKEN_FLOOR",
    "ArchitectureInferenceState",
    "StepZeroActionMode",
    "StepZeroArchitectureConfig",
    "StepZeroArchitectureDirectClient",
    "StepZeroCompletionMode",
    "StepZeroDiagnostics",
    "StepZeroReasoningMode",
    "StepZeroTaskBinding",
]
