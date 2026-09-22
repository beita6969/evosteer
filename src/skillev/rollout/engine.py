"""Atomic on-policy two-pass rollout collection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, NoReturn

from skillev.contracts import JsonValue, TrajectoryStep, stable_hash
from skillev.contracts.action_wire import NATIVE_TOOL_WIRES
from skillev.contracts.skill_exposure import CATALOG_EXPOSURES
from skillev.diagnostics.rollout_progress import current_progress, progress_stage
from skillev.diagnostics.rollout_trace import (
    NullRolloutTraceSink,
    RolloutTraceEvent,
    RolloutTraceSink,
    RolloutTraceStage,
)
from skillev.diagnostics.skill_visibility import CatalogVisibility
from skillev.policy.interface import (
    THINKING_ROLLOUT_PROMPT_ENCODER_VERSION,
    ModelInputWindow,
    PhaseContextSpec,
    encode_policy_prompt,
)
from skillev.runtime import (
    BoundedAgent,
    BoundedAgentState,
    BoundedAgentTurnRequest,
    BudgetLedger,
    BudgetReservation,
    BudgetSettlement,
    BudgetVector,
    EnvironmentSkillInvocationMismatchError,
    EventType,
    RuntimeEventEmitter,
)
from skillev.scoring import render_forward_prefix_from_parts, render_reasoning_prefix

from .action_contract import ActionContract
from .artifact import (
    ActionDraft,
    CompletedStepDraft,
    ReasoningDraft,
    RolloutArtifact,
    RolloutManifest,
    finalize_trajectory_record,
    materialize_trajectory_step,
)
from .codec import ActionCodec, decode_action_segment, decode_reasoning_segment
from .context import AssembledInitialContext, InitialContextAssembler
from .environment import (
    NoSubmissionReason,
    NoTerminalSubmission,
    SubmittedTerminalValue,
    TerminalEvaluationInput,
    TerminalEvaluationRequest,
    TerminalEvaluator,
)
from .errors import (
    RolloutBoundaryError,
    RolloutInfrastructureError,
    RolloutInfrastructureFailure,
    RolloutInfrastructureKind,
)
from .generator import (
    PolicySnapshotMismatchError,
    RolloutGenerationRequest,
    RolloutGenerationResult,
    RolloutGenerator,
)
from .native_wire import NATIVE_TOOL_BOUNDARY_VERSION, NativeToolWire
from .prompt_profiles import validate_initial_context_profile
from .provisional import ProvisionalStep, ProvisionalStepSink
from .types import (
    GenerationPhase,
    PolicySnapshot,
    RolloutRequest,
    RolloutTermination,
    derive_generation_seed,
)

if TYPE_CHECKING:
    from skillev.training.rollout_workflow import RolloutWorkflowResources


class RolloutEngine:
    """Collect exactly one complete trajectory or reject the whole attempt."""

    def __init__(
        self,
        *,
        generator: RolloutGenerator,
        context_assembler: InitialContextAssembler,
        action_codec: ActionCodec,
        bounded_agent: BoundedAgent,
        terminal_evaluator: TerminalEvaluator,
        ledger: BudgetLedger,
        reasoning_call_maximum: BudgetVector,
        action_call_maximum: BudgetVector,
        emitter: RuntimeEventEmitter,
        clock: Callable[[], str],
        workflow_resources: RolloutWorkflowResources | None = None,
        trace_sink: RolloutTraceSink | None = None,
        provisional_sink: ProvisionalStepSink | None = None,
    ) -> None:
        if reasoning_call_maximum.model_calls != 1:
            raise ValueError("reasoning reservation must cover one model call")
        if reasoning_call_maximum.agent_turns != 0:
            raise ValueError("reasoning reservation cannot consume an agent turn")
        if action_call_maximum.model_calls != 1:
            raise ValueError("action reservation must cover one model call")
        if action_call_maximum.agent_turns != 1:
            raise ValueError("action reservation must cover one agent turn")
        self._provisional_sink = provisional_sink
        self._generator = generator
        self._context_assembler = context_assembler
        self._action_codec = action_codec
        self._bounded_agent = bounded_agent
        self._terminal_evaluator = terminal_evaluator
        self._ledger = ledger
        self._reasoning_call_maximum = reasoning_call_maximum
        self._action_call_maximum = action_call_maximum
        self._emitter = emitter
        self._clock = clock
        self._workflow_resources = workflow_resources
        self._trace_sink = trace_sink or NullRolloutTraceSink()

    async def run(self, request: RolloutRequest) -> RolloutArtifact:
        """Run reasoning→action→execution from a fresh in-memory state."""

        started_at = self._clock()
        validate_initial_context_profile(
            profile=request.initial_context_profile,
            active_skill_ids=request.active_skill_ids,
            retrieved_skill_ids=tuple(
                skill.metadata.skill_id for skill in request.retrieved_skills
            ),
        )
        pinned = self._generator.snapshot()
        if pinned.tokenizer_id != self._generator.tokenizer.tokenizer_id:
            self._reject(
                request,
                kind=RolloutInfrastructureKind.POLICY_SNAPSHOT_MISMATCH,
                stage="pin-policy",
                step_index=None,
                message="generator tokenizer does not match its policy snapshot",
            )
        if (
            request.task.environment_id != self._bounded_agent.environment_id
            or request.task.task_family != self._bounded_agent.task_family
        ):
            self._reject(
                request,
                kind=RolloutInfrastructureKind.INITIAL_CONTEXT_MISMATCH,
                stage="bind-environment",
                step_index=None,
                message="rollout task does not match the execution environment",
            )
        assembled = self._context_assembler.assemble(
            decoding=request.decoding,
            task=request.task,
            retrieved_skills=request.retrieved_skills,
            active_skill_ids=request.active_skill_ids,
            library_version=request.library_version,
            tokenizer=self._generator.tokenizer,
            profile=request.initial_context_profile,
        )
        await self._trace_sink.record(
            RolloutTraceEvent(
                request.trajectory_id,
                request.task.task_id,
                None,
                RolloutTraceStage.EPISODE_STARTED,
                {
                    "active_skill_ids": list(assembled.contract.active_skill_ids),
                    "condition_id": request.condition_id,
                    "initial_context": assembled.text,
                    "initial_context_profile": request.initial_context_profile.value,
                    "library_version": request.library_version,
                    "policy_snapshot_id": pinned.snapshot_id,
                    "retrieved_skill_ids": list(assembled.contract.retrieved_skill_ids),
                },
            )
        )
        self._generator.begin_episode(request.trajectory_id, pinned.snapshot_id)
        try:
            return await self._run_started_episode(
                request=request,
                assembled=assembled,
                pinned=pinned,
                started_at=started_at,
            )
        finally:
            self._generator.end_episode(request.trajectory_id)

    async def _run_started_episode(
        self,
        *,
        request: RolloutRequest,
        assembled: AssembledInitialContext,
        pinned: PolicySnapshot,
        started_at: str,
    ) -> RolloutArtifact:
        phase_spec, _ = PhaseContextSpec.split(assembled.text)
        action_codec = self._action_codec
        if phase_spec is not None and phase_spec.action_wire in NATIVE_TOOL_WIRES:
            action_codec = NativeToolWire(
                ActionContract.freeze(
                    request.task.action_surface,
                    retrieved_skill_ids=assembled.contract.retrieved_skill_ids,
                    active_skill_ids=assembled.contract.active_skill_ids,
                ),
                format_version=phase_spec.action_wire,
            )
            if request.decoding.action_boundary_version != NATIVE_TOOL_BOUNDARY_VERSION:
                raise ValueError("native wire requires the declared native stop condition")
        state = BoundedAgentState(invocation_id=request.trajectory_id)
        completed_steps: tuple[TrajectoryStep, ...] = ()
        reasoning_token_counts: tuple[int, ...] = ()
        reasoning_finish_reasons: tuple[str, ...] = ()
        action_finish_reasons: tuple[str, ...] = ()
        stopped_without_action = False
        input_window = ModelInputWindow.from_meta(assembled.contract.meta)
        self._emitter.emit(
            EventType.ROLLOUT_STARTED,
            {
                "assembled_hash": assembled.contract.assembled_hash,
                "decoding_snapshot_id": request.decoding.snapshot_id,
                "library_version": request.library_version,
                "trajectory_id": request.trajectory_id,
            },
        )
        self._emitter.emit(
            EventType.ROLLOUT_POLICY_PINNED,
            {
                "policy_snapshot": pinned.to_value(),
                "trajectory_id": request.trajectory_id,
            },
        )

        visibility = (
            CatalogVisibility.from_skills(request.retrieved_skills)
            if assembled.contract.meta.get("skill_exposure") in CATALOG_EXPOSURES
            else None
        )
        skill_input_evidence: list[dict[str, JsonValue]] = []
        while not state.completed and len(completed_steps) < self._bounded_agent.max_turns:
            step_index = len(completed_steps) + 1
            progress_stage("reasoning-prefix", turn_index=step_index)
            self._require_current_snapshot(request, pinned, step_index=step_index)
            reasoning_prompt = render_reasoning_prefix(
                assembled.text,
                completed_steps,
                step_index,
            )
            reasoning_input = encode_policy_prompt(
                self._generator.tokenizer,
                reasoning_prompt.text,
                initial_text=assembled.text,
                window=input_window,
                native_thinking=(
                    request.decoding.prompt_encoder_version
                    == THINKING_ROLLOUT_PROMPT_ENCODER_VERSION
                ),
            )
            reasoning_visibility = (
                visibility.evidence(
                    reasoning_input,
                    self._generator.tokenizer,
                    previous_steps=completed_steps,
                    library_version=request.library_version,
                    step_index=step_index,
                    phase="reasoning",
                )
                if visibility is not None
                else None
            )
            if reasoning_visibility is not None:
                skill_input_evidence.append(reasoning_visibility)
            reasoning_request = RolloutGenerationRequest(
                episode_id=request.trajectory_id,
                library_version=request.library_version,
                turn_index=step_index,
                phase=GenerationPhase.REASONING,
                input_ids=reasoning_input.ids,
                max_new_tokens=request.decoding.max_reasoning_tokens,
                seed=derive_generation_seed(
                    base_seed=request.decoding.base_seed,
                    coordinate=request.sampling_coordinate,
                    step_index=step_index,
                    phase=GenerationPhase.REASONING,
                ),
                decoding_snapshot_id=request.decoding.snapshot_id,
                expected_policy_snapshot_id=pinned.snapshot_id,
            )
            await self._trace_sink.record(
                RolloutTraceEvent(
                    request.trajectory_id,
                    request.task.task_id,
                    step_index,
                    RolloutTraceStage.REASONING_REQUEST,
                    {
                        **(
                            {"skill_input_evidence": reasoning_visibility}
                            if reasoning_visibility is not None
                            else {}
                        ),
                        "max_new_tokens": reasoning_request.max_new_tokens,
                        "prompt_text": reasoning_prompt.text,
                        "seed": reasoning_request.seed,
                    },
                )
            )
            row = current_progress()
            if row is not None:
                row.begin_phase("reasoning", len(reasoning_request.input_ids))
                row.phase_metrics(
                    **(reasoning_visibility or {}),
                    original_input_tokens=reasoning_input.original_tokens,
                    truncated_input_tokens=reasoning_input.removed_tokens,
                    native_thinking=(
                        request.decoding.prompt_encoder_version
                        == THINKING_ROLLOUT_PROMPT_ENCODER_VERSION
                    ),
                )
            reasoning_result = await self._generate(
                request=request,
                generation_request=reasoning_request,
                pinned=pinned,
                step_index=step_index,
            )
            if row is not None:
                row.finish_phase(
                    reasoning_result.usage.output_tokens, reasoning_result.finish_reason
                )
            reasoning = decode_reasoning_segment(
                self._generator.tokenizer,
                reasoning_result.content_token_ids,
            )
            await self._trace_sink.record(
                RolloutTraceEvent(
                    request.trajectory_id,
                    request.task.task_id,
                    step_index,
                    RolloutTraceStage.REASONING_RESULT,
                    {
                        "finish_reason": reasoning_result.finish_reason,
                        "text": reasoning.text,
                        "token_count": len(reasoning.token_ids),
                    },
                )
            )
            reasoning_draft = ReasoningDraft(
                step_index=step_index,
                prompt_text=reasoning_prompt.text,
                prompt_hash=reasoning_prompt.prompt_hash,
                text=reasoning.text,
                generated_token_ids=reasoning.token_ids,
                policy_snapshot_id=reasoning_result.policy_snapshot_id,
            )
            reasoning_finish_reasons = (
                *reasoning_finish_reasons,
                reasoning_result.finish_reason,
            )

            forward_prefix = render_forward_prefix_from_parts(
                assembled.text,
                completed_steps,
                step_index,
                reasoning.text,
            )
            action_input = encode_policy_prompt(
                self._generator.tokenizer,
                forward_prefix.text,
                initial_text=assembled.text,
                window=input_window,
            )
            action_visibility = (
                visibility.evidence(
                    action_input,
                    self._generator.tokenizer,
                    previous_steps=completed_steps,
                    library_version=request.library_version,
                    step_index=step_index,
                    phase="action",
                )
                if visibility is not None
                else None
            )
            if action_visibility is not None:
                skill_input_evidence.append(action_visibility)
            action_request = RolloutGenerationRequest(
                action_boundary_version=request.decoding.action_boundary_version,
                episode_id=request.trajectory_id,
                library_version=request.library_version,
                turn_index=step_index,
                phase=GenerationPhase.ACTION,
                input_ids=action_input.ids,
                max_new_tokens=request.decoding.max_action_tokens,
                seed=derive_generation_seed(
                    base_seed=request.decoding.base_seed,
                    coordinate=request.sampling_coordinate,
                    step_index=step_index,
                    phase=GenerationPhase.ACTION,
                ),
                decoding_snapshot_id=request.decoding.snapshot_id,
                expected_policy_snapshot_id=pinned.snapshot_id,
            )
            await self._trace_sink.record(
                RolloutTraceEvent(
                    request.trajectory_id,
                    request.task.task_id,
                    step_index,
                    RolloutTraceStage.ACTION_REQUEST,
                    {
                        **(
                            {"skill_input_evidence": action_visibility}
                            if action_visibility is not None
                            else {}
                        ),
                        "max_new_tokens": action_request.max_new_tokens,
                        "prompt_text": forward_prefix.text,
                        "seed": action_request.seed,
                    },
                )
            )
            if row is not None:
                row.begin_phase("action", len(action_request.input_ids))
                row.phase_metrics(
                    native_thinking=False,
                    reasoning_condition="observed-c_t-response-ended",
                    reasoning_finish_reason=reasoning_result.finish_reason,
                    **(action_visibility or {}),
                    original_input_tokens=action_input.original_tokens,
                    truncated_input_tokens=action_input.removed_tokens,
                )
            action_result = await self._generate(
                request=request,
                generation_request=action_request,
                pinned=pinned,
                step_index=step_index,
            )
            if row is not None:
                row.finish_phase(action_result.usage.output_tokens, action_result.finish_reason)
            # An immediately sampled stop token is a model outcome, not an
            # infrastructure failure.  Preserve that exact sampled token as a
            # one-token edge so K_t remains positive and teacher forcing can
            # score the probability of stopping without a submission.  A
            # response containing no sampled token at all is still malformed.
            stop_only_action = not action_result.content_token_ids and bool(
                action_result.stop_token_ids
            )
            action_token_ids = (
                action_result.stop_token_ids
                if stop_only_action
                else action_result.content_token_ids
            )
            if not action_token_ids:
                self._reject(
                    request,
                    kind=RolloutInfrastructureKind.EMPTY_ACTION,
                    stage="decode-action",
                    step_index=step_index,
                    message="action generation produced no sampled tokens",
                )
            try:
                action_segment = decode_action_segment(
                    self._generator.tokenizer,
                    action_token_ids,
                )
            except RolloutBoundaryError:
                self._reject(
                    request,
                    kind=RolloutInfrastructureKind.EMPTY_ACTION,
                    stage="decode-action",
                    step_index=step_index,
                    message="action generation produced no model-visible text",
                )
            parse_result = action_codec.parse(action_segment.text)
            from skillev.diagnostics.action_submission import ActionSubmissionOutcome

            submission = ActionSubmissionOutcome.observe(
                action_segment.text,
                parse_result,
                finish_reason=action_result.finish_reason,
                output_tokens=action_result.usage.output_tokens,
                action_token_cap=request.decoding.max_action_tokens,
                turns_remaining=self._bounded_agent.max_turns - step_index,
            )
            phase_spec, _ = PhaseContextSpec.split(assembled.text)
            await self._trace_sink.record(
                RolloutTraceEvent(
                    request.trajectory_id,
                    request.task.task_id,
                    step_index,
                    RolloutTraceStage.ACTION_RESULT,
                    {
                        "submission_outcome": submission.to_value(),
                        "finish_reason": action_result.finish_reason,
                        "raw_text": action_segment.text,
                        "token_count": len(action_segment.token_ids),
                    },
                )
            )
            await self._trace_sink.record(
                RolloutTraceEvent(
                    request.trajectory_id,
                    request.task.task_id,
                    step_index,
                    RolloutTraceStage.ACTION_PARSED,
                    {
                        "action": (
                            None if parse_result.action is None else parse_result.action.to_value()
                        ),
                        "public_error_code": parse_result.public_error_code,
                        "raw_text": action_segment.text,
                        "status": parse_result.status.value,
                    },
                )
            )
            action_draft = ActionDraft(
                step_index=step_index,
                forward_prefix_text=forward_prefix.text,
                forward_prefix_hash=forward_prefix.prefix_hash,
                text=action_segment.text,
                token_ids=action_segment.token_ids,
                parse_result=parse_result,
                policy_snapshot_id=action_result.policy_snapshot_id,
            )
            action_finish_reasons = (*action_finish_reasons, action_result.finish_reason)
            progress_stage("environment-execute")
            try:
                turn = await self._bounded_agent.execute_turn(
                    state,
                    BoundedAgentTurnRequest(
                        trajectory_id=request.trajectory_id,
                        step_index=step_index,
                        action_text=action_segment.text,
                        action_token_ids=action_segment.token_ids,
                        parse_result=parse_result,
                        retrieved_skill_ids=assembled.contract.retrieved_skill_ids,
                        active_skill_ids=assembled.contract.active_skill_ids,
                        public_submission_feedback=submission.public_feedback()
                        if phase_spec is not None and phase_spec.submission_feedback
                        else None,
                    ),
                )
            except EnvironmentSkillInvocationMismatchError:
                self._reject(
                    request,
                    kind=RolloutInfrastructureKind.ENVIRONMENT_SKILL_INVOCATION_MISMATCH,
                    stage="execute-action",
                    step_index=step_index,
                    message="environment skill credit differs from the structured action",
                )
            submission = replace(
                submission,
                admitted=turn.admitted,
                executed=turn.executed,
                execution_status=turn.observation.observation_status if turn.executed else None,
            )
            if row is not None:
                row.submission_outcome(dict(submission.to_value()))
            step = materialize_trajectory_step(
                initial_text=assembled.text,
                previous_steps=completed_steps,
                draft=CompletedStepDraft(
                    reasoning=reasoning_draft,
                    action=action_draft,
                    observation=turn.observation,
                ),
            )
            from skillev.diagnostics.action_failures import (
                ACTION_FAILURE_FORMAT,
                classify_action_outcome,
            )

            observed_value = turn.observation.public_value
            observed_error = (
                observed_value.get("error") if isinstance(observed_value, dict) else None
            )
            action_diagnostics = classify_action_outcome(
                action_segment.text,
                parse_status=parse_result.status.value,
                finish_reason=action_result.finish_reason,
                action_kind=None if parse_result.action is None else parse_result.action.kind.value,
                completed=turn.state.completed,
                observation_status=step.observation_status,
                action_wire=action_codec.format_version,
                public_error_code=(
                    observed_error
                    if isinstance(observed_error, str)
                    else parse_result.public_error_code
                ),
                available_native_names=(
                    tuple(binding.name for binding in action_codec.bindings)
                    if isinstance(action_codec, NativeToolWire)
                    else None
                ),
                visible_skill_ids=assembled.contract.retrieved_skill_ids,
                output_token_count=len(action_result.content_token_ids),
            )
            if row is not None:
                row.action_outcome(action_diagnostics)
            await self._trace_sink.record(
                RolloutTraceEvent(
                    request.trajectory_id,
                    request.task.task_id,
                    step_index,
                    RolloutTraceStage.ENVIRONMENT_RESULT,
                    {
                        "submission_outcome": submission.to_value(),
                        "action_diagnostics": {
                            "format": ACTION_FAILURE_FORMAT,
                            "action_wire": action_codec.format_version,
                            "labels": list(action_diagnostics),
                            "admitted": None,
                            "executed": None,
                            "terminal_success": None,
                        },
                        "budget_usage": turn.observation.budget_usage.to_value(),
                        "invoked_skill_ids": list(turn.observation.invoked_skill_ids),
                        "observation": turn.observation.public_value,
                        "observation_status": turn.observation.observation_status,
                    },
                )
            )
            if self._provisional_sink is not None:
                progress_stage("provisional-admission")
                await self._provisional_sink(
                    ProvisionalStep(
                        request.trajectory_id,
                        request.task.task_id,
                        pinned,
                        request.library_version,
                        assembled.contract.query,
                        assembled.text,
                        completed_steps,
                        step,
                        input_window,
                        action_input,
                    )
                )
            progress_stage("step-materialized")
            completed_steps = (*completed_steps, step)
            reasoning_token_counts = (
                *reasoning_token_counts,
                len(reasoning.token_ids),
            )
            state = turn.state
            self._emitter.emit(
                EventType.ROLLOUT_STEP_COMMITTED,
                {
                    "action_token_count": step.action_token_count,
                    "forward_prefix_hash": step.forward_prefix_hash,
                    "hindsight_prefix_hash": step.hindsight_prefix_hash,
                    "observation_status": step.observation_status,
                    "step_index": step.index,
                    "trajectory_id": request.trajectory_id,
                },
            )
            if stop_only_action:
                stopped_without_action = True
                break

        self._require_current_snapshot(request, pinned, step_index=None)
        evaluation_input: TerminalEvaluationInput
        if state.completed:
            termination = RolloutTermination.COMPLETED
            evaluation_input = SubmittedTerminalValue(state.completion_value)
        elif stopped_without_action:
            termination = RolloutTermination.NO_VALID_COMPLETE_ACTION
            evaluation_input = NoTerminalSubmission(NoSubmissionReason.NO_VALID_COMPLETE_ACTION)
        else:
            termination = RolloutTermination.HORIZON_EXHAUSTED
            evaluation_input = NoTerminalSubmission(NoSubmissionReason.HORIZON_EXHAUSTED)
        from .environment import TerminalActionEvidence

        last_step = completed_steps[-1]
        evaluation_request = TerminalEvaluationRequest(
            trajectory_id=request.trajectory_id,
            task_id=request.task.task_id,
            termination=termination,
            evaluation_input=evaluation_input,
            last_action=TerminalActionEvidence(
                last_step.index,
                last_step.action_text,
                action_codec.parse(last_step.action_text).status.value,
                last_step.observation_status,
                action_finish_reasons[-1],
            ),
            public_transcript_hash=stable_hash(
                {
                    "initial_context_hash": assembled.contract.assembled_hash,
                    "steps": [step.to_value() for step in completed_steps],
                }
            ),
        )
        await self._trace_sink.record(
            RolloutTraceEvent(
                request.trajectory_id,
                request.task.task_id,
                None,
                RolloutTraceStage.TERMINAL_REQUEST,
                {
                    "submission_produced": isinstance(evaluation_input, SubmittedTerminalValue),
                    "termination": termination.value,
                },
            )
        )
        progress_stage("terminal-evaluation")
        if self._workflow_resources is None:
            reward = await self._terminal_evaluator.evaluate(evaluation_request)
        else:
            async with self._workflow_resources.terminal_evaluations.lease():
                reward = await self._terminal_evaluator.evaluate(evaluation_request)
        await self._trace_sink.record(
            RolloutTraceEvent(
                request.trajectory_id,
                request.task.task_id,
                None,
                RolloutTraceStage.TERMINAL_RESULT,
                {
                    "native_metric_names": [reward.native_metric_name],
                    "posterior_success": reward.success,
                    "reward": reward.value,
                    "verifier_version": reward.verifier_version,
                },
            )
        )

        completed_at = self._clock()
        record = finalize_trajectory_record(
            request=request,
            assembled=assembled,
            steps=completed_steps,
            reward=reward,
            tokenizer=self._generator.tokenizer,
            created_at=completed_at,
        )
        artifact = RolloutArtifact(
            initial_context=assembled,
            record=record,
            skill_input_evidence=tuple(skill_input_evidence) if visibility is not None else None,
            manifest=RolloutManifest(
                trajectory_id=request.trajectory_id,
                task_id=request.task.task_id,
                policy_snapshot=pinned,
                library_version=request.library_version,
                sampling_coordinate=request.sampling_coordinate,
                decoding_snapshot_id=request.decoding.snapshot_id,
                assembler_version=self._context_assembler.assembler_version,
                action_format_version=action_codec.format_version,
                generator_backend_id=pinned.backend_id,
                termination=termination,
                reasoning_token_counts=reasoning_token_counts,
                started_at=started_at,
                completed_at=completed_at,
                prompt_encoder_version=request.decoding.prompt_encoder_version,
                action_boundary_version=request.decoding.action_boundary_version,
                reasoning_finish_reasons=reasoning_finish_reasons,
                action_finish_reasons=action_finish_reasons,
                condition_id=request.condition_id,
                initial_context_profile=request.initial_context_profile.value,
            ),
        )
        self._emitter.emit(EventType.TERMINAL_REWARD_RECORDED, reward.to_value())
        self._emitter.emit(EventType.ROLLOUT_COMPLETED, artifact.to_value())
        return artifact

    def _require_current_snapshot(
        self,
        request: RolloutRequest,
        pinned: PolicySnapshot,
        *,
        step_index: int | None,
    ) -> None:
        if self._generator.snapshot() != pinned:
            self._reject(
                request,
                kind=RolloutInfrastructureKind.POLICY_SNAPSHOT_MISMATCH,
                stage="verify-policy-snapshot",
                step_index=step_index,
                message="generator policy snapshot changed during rollout",
            )

    async def _generate(
        self,
        *,
        request: RolloutRequest,
        generation_request: RolloutGenerationRequest,
        pinned: PolicySnapshot,
        step_index: int,
    ) -> RolloutGenerationResult:
        phase = generation_request.phase
        reservation = BudgetReservation(
            reservation_id=f"{request.trajectory_id}:{step_index}:{phase.value}:model",
            run_id=self._ledger.run_id,
            attempt_id=self._ledger.attempt_id,
            invocation_id=request.trajectory_id,
            maximum=(
                self._reasoning_call_maximum
                if phase is GenerationPhase.REASONING
                else self._action_call_maximum
            ),
        )
        if len(generation_request.input_ids) > reservation.maximum.input_tokens:
            # Refuse before dispatch: no hidden truncation and no unknown terminal
            # outcome relabeled as a failed task. The whole batch remains uncommitted.
            raise ValueError("model input exceeds the declared per-request token allowance")
        self._ledger.reserve(reservation)
        self._emitter.emit(
            EventType.BUDGET_RESERVED,
            {
                "maximum": reservation.maximum.to_value(),
                "reservation_id": reservation.reservation_id,
            },
        )
        try:
            if self._workflow_resources is None:
                result = await self._generator.generate(generation_request)
            else:
                endpoint = getattr(self._generator, "execution_endpoint", None)
                limiter = (
                    self._workflow_resources.model_requests
                    if endpoint is None
                    else self._workflow_resources.model_limiter(endpoint(generation_request))
                )
                async with limiter.lease(
                    token_cost=len(generation_request.input_ids)
                    + generation_request.max_new_tokens,
                    role="actor",
                ):
                    result = await self._generator.generate(generation_request)
        except PolicySnapshotMismatchError:
            self._reject(
                request,
                kind=RolloutInfrastructureKind.POLICY_SNAPSHOT_MISMATCH,
                stage=f"generate-{phase.value}",
                step_index=step_index,
                message="generator rejected the exact pinned policy snapshot",
            )

        actual_usage = result.usage.add(
            BudgetVector(
                agent_turns=(1 if generation_request.phase is GenerationPhase.ACTION else 0),
            )
        )
        if result.usage.input_tokens != len(generation_request.input_ids):
            raise ValueError("generator input-token usage differs from the request")
        if result.usage.model_calls != 1:
            raise ValueError("one generation request must report one model call")
        settlement = BudgetSettlement(
            reservation_id=reservation.reservation_id,
            actual=actual_usage,
        )
        self._ledger.settle(settlement)
        self._emitter.emit(
            EventType.BUDGET_SETTLED,
            {
                "actual": actual_usage.to_value(),
                "reservation_id": reservation.reservation_id,
            },
        )
        if result.policy_snapshot_id != pinned.snapshot_id:
            self._reject(
                request,
                kind=RolloutInfrastructureKind.POLICY_SNAPSHOT_MISMATCH,
                stage=f"verify-{phase.value}-policy",
                step_index=step_index,
                message="generator returned another policy snapshot",
            )
        if result.backend_id != pinned.backend_id:
            self._reject(
                request,
                kind=RolloutInfrastructureKind.POLICY_SNAPSHOT_MISMATCH,
                stage=f"verify-{phase.value}-backend",
                step_index=step_index,
                message="generator returned another backend identity",
            )
        return result

    def _reject(
        self,
        request: RolloutRequest,
        *,
        kind: RolloutInfrastructureKind,
        stage: str,
        step_index: int | None,
        message: str,
    ) -> NoReturn:
        failure = RolloutInfrastructureFailure(
            trajectory_id=request.trajectory_id,
            task_id=request.task.task_id,
            kind=kind,
            stage=stage,
            step_index=step_index,
            public_message=message,
        )
        self._emitter.emit(EventType.ROLLOUT_REJECTED, failure.to_value())
        raise RolloutInfrastructureError(failure)
