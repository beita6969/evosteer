"""Evaluation-only generation with explicit budgets and reproducible call substreams."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace
from typing import Protocol, runtime_checkable

from skillev.rollout import GenerationPhase, RolloutGenerationResult
from skillev.rollout.evaluation_sglang import (
    EvaluationGenerationConstraint,
    EvaluationGenerationProfile,
    EvaluationRolloutGenerator,
)
from skillev.runtime import BudgetVector

from .direct_baseline import DirectGenerationRequest
from .native_channels import ChannelStatus, split_native_channels
from .native_continuation import (
    NativeContinuation,
    native_call_allowance,
    thinking_boundary_at_reserve,
)
from .native_tool_calls import NativeTools
from .public_context import PackedPrompt, PublicPrompt, PublicView
from .sampling_stream import SAMPLING_SEED_SCHEDULE, evaluation_call_seed
from .sealed_candidates import EventRecorder
from .step0_completion import evaluation_request
from .step0_integrity import InferenceArm, InterventionCounts
from .submission_outcome import SubmissionBudget


@runtime_checkable
class IntegrityTokenizer(Protocol):
    def encode_integrity_messages(
        self,
        messages: tuple[dict[str, str], ...],
        *,
        enable_thinking: bool,
        tools: NativeTools = (),
    ) -> list[int]: ...


class EvaluationBudgetExhausted(RuntimeError):  # noqa: N818 -- evaluation budget status
    """The evaluated actor spent its declared budget; not an infrastructure outage."""


class EvaluationDeadlineExceeded(EvaluationBudgetExhausted):
    """No new call/action was admitted after the episode's soft deadline."""


@dataclass(frozen=True, slots=True)
class CallAccounting:
    call_id: str
    participant: str
    purpose: str
    transport_status: str
    input_tokens: int | None = None
    output_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.transport_status not in {
            "started",
            "completed",
            "failed",
            "unknown",
            "not-admitted",
        }:
            raise ValueError("unknown model call status")
        for value in (self.input_tokens, self.output_tokens):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("model token usage must be nonnegative or unknown")
        if self.transport_status == "completed" and (
            self.input_tokens is None or self.output_tokens is None
        ):
            raise ValueError("completed calls require actual usage")


def with_framework_instruction(
    messages: tuple[dict[str, str], ...], instruction: str
) -> tuple[dict[str, str], ...]:
    """Keep framework controls separate from user dialogue and untrusted returns."""
    # Qwen's native template accepts only one system message, at index zero.
    # Add controls to that trusted message without changing any dialogue or
    # elevating tool/peer content into system instructions.
    if messages and messages[0]["role"] == "system":
        instruction = messages[0]["content"] + "\n\n" + instruction
        messages = messages[1:]
    return ({"role": "system", "content": instruction}, *messages)


@dataclass(slots=True)
class IntegrityGeneration:
    generator: EvaluationRolloutGenerator
    arm: InferenceArm
    counts: InterventionCounts
    journal: EventRecorder | None = None
    run_id: str = "unpublished-development"
    call_limit: int | None = None
    output_token_limit: int | None = None
    context_length: int | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    calls: list[CallAccounting] = field(default_factory=list)
    visible_event_ids: tuple[str, ...] = ()
    native_chunk_tokens: int | None = None
    native_final_reserve_tokens: int = 0
    native_close_at_reserve: bool = False
    finalization_reserve_tokens: int = 0
    deadline_monotonic: float | None = None
    _last_packed_prompt: PackedPrompt | None = field(default=None, init=False)

    def trace(self, episode_id: str, stage: str, payload: object) -> None:
        if self.journal is not None:
            self.journal.record((self.run_id, self.arm.arm_id, episode_id), stage, payload)

    def require_time_remaining(self) -> None:
        if self.deadline_monotonic is not None and time.monotonic() >= self.deadline_monotonic:
            raise EvaluationDeadlineExceeded("episode admission deadline reached")

    async def call(
        self,
        request: DirectGenerationRequest,
        episode_id: str,
        messages: tuple[dict[str, str], ...] | PublicPrompt,
        *,
        phase: GenerationPhase,
        maximum_tokens: int,
        maximum_calls: int | None = None,
        constraint: EvaluationGenerationConstraint | None = None,
        participant: str = "owner",
        purpose: str = "decision",
        public_view: PublicView | None = None,
        tools: NativeTools = (),
    ) -> tuple[str, str, RolloutGenerationResult]:
        continuation = None
        previous_channel_status = None
        previous_boundary = None
        usage = BudgetVector()
        calls_started = self.counts.model_calls
        enabled = self.arm.native_thinking and self.native_chunk_tokens is not None
        context_output_tokens = maximum_tokens
        if enabled and self.output_token_limit is not None:
            context_output_tokens = self.output_token_limit - self.output_tokens
            call_bounds = []
            if maximum_calls is not None:
                call_bounds.append(maximum_calls)
            if self.call_limit is not None:
                call_bounds.append(self.call_limit - self.counts.model_calls)
            if call_bounds:
                # An interactive episode spans many independently packed actions.
                # Reserve this response's possible continuation, not every future
                # action's output. The shared episode ledger remains unchanged.
                per_call = min(maximum_tokens, self.native_chunk_tokens or maximum_tokens)
                context_output_tokens = min(context_output_tokens, per_call * min(call_bounds))
        if enabled:
            prompt = (
                messages if isinstance(messages, PublicPrompt) else PublicPrompt.required(messages)
            )
            messages = prompt.with_instruction(
                "Native thinking uses one continuous token stream. A length pause resumes that "
                "same draft, not an independent solution. The episode output allowance is "
                f"{self.output_token_limit} tokens, including a final "
                f"{self.native_final_reserve_tokens} "
                "tokens reserved to finish thinking and submit your answer. No draft number is "
                "submitted before your actual final channel."
                + (
                    " If thinking is still open at the reserved boundary, the serving layer "
                    "closes only the thinking channel. Use the remaining allowance for your "
                    "own concise final answer, not another full derivation."
                    if self.native_close_at_reserve
                    else ""
                )
            )
        while True:
            limit = maximum_tokens
            if (
                enabled
                and self.output_token_limit is not None
                and self.native_chunk_tokens is not None
            ):
                limit = min(
                    limit,
                    native_call_allowance(
                        self.output_token_limit - self.output_tokens,
                        self.native_chunk_tokens,
                        self.native_final_reserve_tokens,
                    ),
                )
            boundary = thinking_boundary_at_reserve(
                enabled=self.native_close_at_reserve,
                continuing=continuation is not None,
                remaining=(self.output_token_limit or 0) - self.output_tokens,
                reserve=self.native_final_reserve_tokens,
                channel_status=previous_channel_status,
                previous_boundary=previous_boundary,
            )
            if boundary is not None:
                limit = 1
            text, thought, result = await self._call_once(
                request,
                episode_id,
                messages,
                phase=phase,
                maximum_tokens=limit,
                constraint=constraint,
                participant=participant,
                purpose=purpose,
                public_view=public_view,
                tools=tools,
                continuation=continuation,
                context_output_tokens=context_output_tokens,
                thinking_boundary=boundary,
            )
            usage = usage.add(result.usage)
            tokens = (
                () if continuation is None else continuation.output_token_ids
            ) + result.content_token_ids
            channels = split_native_channels(
                tokens,
                enabled=self.arm.native_thinking,
                tokenizer=self.generator.tokenizer,
                finish_reason=result.finish_reason,
            )
            previous_channel_status = channels.status
            previous_boundary = boundary
            remaining = (
                None
                if self.output_token_limit is None
                else self.output_token_limit - self.output_tokens
            )
            if not (
                enabled
                and result.finish_reason == "length"
                and channels.status
                in {
                    ChannelStatus.REASONING_UNFINISHED,
                    ChannelStatus.FINAL_EMPTY,
                    ChannelStatus.FINAL_UNFINISHED,
                }
                and remaining is not None
                and remaining > 0
                and (self.call_limit is None or self.counts.model_calls < self.call_limit)
                and (
                    maximum_calls is None or self.counts.model_calls - calls_started < maximum_calls
                )
            ):
                # This is a controller accounting view over one token stream.
                # The journal retains each actual serving chunk and its own usage.
                return text, thought, replace(result, content_token_ids=tokens, usage=usage)
            if self._last_packed_prompt is None:
                raise RuntimeError("native continuation has no original public prompt")
            continuation = NativeContinuation(
                self.calls[-1].call_id, self._last_packed_prompt, tokens
            )

    async def _call_once(
        self,
        request: DirectGenerationRequest,
        episode_id: str,
        messages: tuple[dict[str, str], ...] | PublicPrompt,
        *,
        phase: GenerationPhase,
        maximum_tokens: int,
        constraint: EvaluationGenerationConstraint | None = None,
        participant: str = "owner",
        purpose: str = "decision",
        public_view: PublicView | None = None,
        tools: NativeTools = (),
        continuation: NativeContinuation | None = None,
        context_output_tokens: int | None = None,
        thinking_boundary: str | None = None,
    ) -> tuple[str, str, RolloutGenerationResult]:
        self.require_time_remaining()
        if self.call_limit is not None and self.counts.model_calls >= self.call_limit:
            raise EvaluationBudgetExhausted("model call budget exhausted")
        if self.output_token_limit is not None:
            budget = SubmissionBudget(self.output_token_limit, self.finalization_reserve_tokens)
            maximum_tokens = min(
                maximum_tokens, budget.allowance(self.output_tokens, self.counts.model_calls)
            )
        if maximum_tokens < 1:
            raise EvaluationBudgetExhausted("output token budget exhausted")
        prompt = messages if isinstance(messages, PublicPrompt) else PublicPrompt.required(messages)
        if self.call_limit is not None and self.output_token_limit is not None:
            # Update the existing interactive runtime surface, not the opening
            # system prefix on every action. Without that surface, preserve the
            # conversational system controls; never add a fake tool/user turn.
            prompt = prompt.with_runtime_metadata(
                "Remaining system-wide budget: "
                f"{self.call_limit - self.counts.model_calls} model calls and "
                f"{self.output_token_limit - self.output_tokens} output tokens "
                "across your calls in this episode. "
                f"This response can use at most {maximum_tokens} output tokens, including "
                "any explanation and submission syntax. Finish the answer or tool call "
                "within that limit. These are upper limits, not required work."
                + (
                    f" The first response leaves {self.finalization_reserve_tokens} tokens "
                    "inside that same total for your own final submission "
                    "or interface clarification."
                    if self.finalization_reserve_tokens and self.counts.model_calls == 0
                    else ""
                )
            )
        direct = request.profile
        profile = EvaluationGenerationProfile(
            profile_id=f"{direct.profile_id}:{self.arm.arm_id}:{phase.value}",
            enable_thinking=self.arm.native_thinking,
            temperature=direct.temperature,
            top_p=direct.top_p,
            top_k=direct.top_k,
            min_p=direct.min_p,
            presence_penalty=direct.presence_penalty,
            repetition_penalty=direct.repetition_penalty,
            max_new_tokens=maximum_tokens,
            seed=evaluation_call_seed(
                direct.seed, self.counts.model_calls, sampling_mode=direct.sampling_mode
            ),
            stop=direct.stop,
            sampling_mode=direct.sampling_mode,
            thinking_boundary=thinking_boundary,
        )
        tokenizer = self.generator.tokenizer
        if not isinstance(tokenizer, IntegrityTokenizer):
            raise TypeError("clean evaluation requires the neutral public-message tokenizer")

        def encode(rows: tuple[dict[str, str], ...]) -> list[int]:
            if tools:
                return tokenizer.encode_integrity_messages(
                    rows, enable_thinking=self.arm.native_thinking, tools=tools
                )
            return tokenizer.encode_integrity_messages(
                rows, enable_thinking=self.arm.native_thinking
            )

        packed = (
            continuation.prompt
            if continuation is not None
            else prompt.pack(
                encode,
                context_length=self.context_length,
                maximum_output_tokens=(
                    maximum_tokens if context_output_tokens is None else context_output_tokens
                ),
            )
        )
        input_ids = continuation.input_token_ids if continuation is not None else packed.input_ids
        if (
            self.context_length is not None
            and len(input_ids) + maximum_tokens > self.context_length
        ):
            raise ValueError("native continuation exceeds its declared context allowance")
        self._last_packed_prompt = packed
        self.visible_event_ids = packed.visible_event_ids
        # A native thinking turn must be allowed to emit its reasoning channel.
        # Its final action still passes the same explicit-decision parser.
        effective_constraint = None if self.arm.native_thinking else constraint
        self.require_time_remaining()
        self.trace(
            episode_id,
            "rendered-request",
            {
                "messages": packed.messages,
                "tools": tools,
                "profile": asdict(profile),
                "experiment_seed": direct.seed,
                "sampling_seed_schedule": SAMPLING_SEED_SCHEDULE,
                "continuation_of_call_id": continuation.call_id if continuation else None,
                "input_tokens": len(input_ids),
                "visible_event_ids": packed.visible_event_ids,
                "archived_message_count": packed.omitted_messages,
                "skill_ids": tuple(block.skill_id for block in packed.skill_blocks),
                "participant": participant,
                "purpose": purpose,
                "public_view": asdict(public_view) if public_view is not None else None,
                "constraint": asdict(effective_constraint) if effective_constraint else None,
            },
        )
        call_id = f"model-call-{self.counts.model_calls + 1}"
        counts_before_call = asdict(self.counts)
        call_index = len(self.calls)
        self.calls.append(CallAccounting(call_id, participant, purpose, "started"))
        # Admission, including a later timeout/cancellation, spends the call.
        # Unknown serving usage is never silently represented as zero.
        self.counts.model_calls += 1
        proposed_skills = {block.skill_id for block in prompt.blocks if block.skill_id is not None}
        if proposed_skills:
            visible_skills = {
                block.skill_id for block in packed.skill_blocks if block.skill_id is not None
            }
            omitted_skills = sorted(proposed_skills - visible_skills)
            self.counts.skill_blocks_injected += len(packed.skill_blocks)
            self.counts.skill_body_tokens += sum(
                len(tokenizer.encode(block.content)) for block in packed.skill_blocks
            )
            self.counts.skill_context_omissions += len(omitted_skills)
            self.trace(
                episode_id,
                "skill-context",
                {
                    "call_id": call_id,
                    "visible_skill_ids": sorted(visible_skills),
                    "omitted_skill_ids": omitted_skills,
                },
            )
        if participant != "owner":
            self.counts.peer_model_calls += 1
        if purpose == "interface-repair":
            self.counts.communication_repairs += 1
            self.trace(
                episode_id,
                "model-interface-repair",
                {
                    "status": "started",
                    "call_id": call_id,
                    "participant": participant,
                },
            )
        completed = False
        try:
            result = await self.generator.generate_evaluation(
                evaluation_request(
                    input_ids=input_ids,
                    max_new_tokens=maximum_tokens,
                    seed=profile.seed,
                    decoding_profile_id=profile.profile_id,
                    policy_snapshot_id=self.generator.snapshot().snapshot_id,
                    phase=phase,
                ),
                profile=profile,
                constraint=effective_constraint,
            )
            completed = True
        except EvaluationDeadlineExceeded:
            # The broker may reject admission if the deadline crossed during
            # local packing/IPC. This is not a serving attempt or unknown usage.
            completed = True
            for name, value in counts_before_call.items():
                setattr(self.counts, name, value)
            self.calls[call_index] = CallAccounting(call_id, participant, purpose, "not-admitted")
            self.trace(episode_id, "call-accounting", asdict(self.calls[call_index]))
            raise
        finally:
            if not completed:
                # Bookkeeping only: the original failure/cancellation propagates
                # unchanged, with no fallback or automatic serving retry.
                self.calls[call_index] = CallAccounting(call_id, participant, purpose, "unknown")
                self.trace(episode_id, "call-accounting", asdict(self.calls[call_index]))
        self.calls[call_index] = CallAccounting(
            call_id,
            participant,
            purpose,
            "completed",
            result.usage.input_tokens,
            result.usage.output_tokens,
        )
        self.trace(episode_id, "call-accounting", asdict(self.calls[call_index]))
        self.input_tokens += result.usage.input_tokens
        self.output_tokens += result.usage.output_tokens
        raw = tokenizer.decode(result.content_token_ids)
        self.trace(
            episode_id,
            "model-response",
            {
                "text": raw,
                "usage": result.usage.to_value(),
                "phase": phase.value,
                "finish_reason": result.finish_reason,
            },
        )
        stream_tokens = (
            () if continuation is None else continuation.output_token_ids
        ) + result.content_token_ids
        channels = split_native_channels(
            stream_tokens,
            enabled=self.arm.native_thinking,
            tokenizer=tokenizer,
            finish_reason=result.finish_reason,
        )
        final = (
            raw
            if channels.final_token_ids == result.content_token_ids
            else tokenizer.decode(channels.final_token_ids)
        )
        if channels.status is not ChannelStatus.COMPLETE:
            final = ""
        thought = (
            tokenizer.decode(channels.reasoning_token_ids) if channels.reasoning_token_ids else ""
        )
        return final, thought, result
