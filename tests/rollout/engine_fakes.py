"""Deterministic, answer-free fakes for real RolloutEngine behavior tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from skillev.contracts import (
    JsonValue,
    ScientificSamplingCoordinate,
    SuccessRule,
    TerminalReward,
    normalize_json,
    stable_hash,
)
from skillev.policy.interface import AdapterRole
from skillev.rollout import (
    CanonicalInitialContextAssembler,
    DecodingSnapshot,
    EnvironmentObservation,
    PolicySnapshot,
    RolloutEngine,
    RolloutGenerationRequest,
    RolloutGenerationResult,
    RolloutRequest,
    RolloutTask,
    StructuredJsonActionCodec,
    TerminalEvaluationRequest,
    TerminalEvaluatorError,
)
from skillev.rollout.prompt_profiles import InitialContextProfile
from skillev.runtime import (
    BoundedAgent,
    BoundedAgentPolicy,
    BudgetLedger,
    BudgetVector,
    FullRetrievedSkillContext,
    LiveAttemptEventLog,
    RetrievalInclusionReason,
    RuntimeEventEmitter,
    SkillMetadata,
    StructuredAction,
)

FIXED_TIME = "2026-07-20T12:00:00.000000Z"
PRIVATE_CANARY = "PRIVATE-EVALUATOR-CANARY-DO-NOT-PROMPT"


class ByteTokenizer:
    """Exact UTF-8 tokenizer with opt-in non-round-tripping decode cases."""

    tokenizer_id = "byte-tokenizer-v1"

    def __init__(
        self,
        *,
        decode_overrides: dict[tuple[int, ...], str] | None = None,
    ) -> None:
        self.decode_overrides = decode_overrides or {}
        self.encoded_texts: list[str] = []
        self.rollout_encoded_texts: list[str] = []
        self.decoded_ids: list[tuple[int, ...]] = []

    def encode(self, text: str) -> list[int]:
        self.encoded_texts.append(text)
        return list(text.encode("utf-8"))

    def encode_rollout_prompt(self, text: str) -> list[int]:
        self.rollout_encoded_texts.append(text)
        return [0, *text.encode("utf-8")]

    def decode(self, token_ids: tuple[int, ...]) -> str:
        self.decoded_ids.append(token_ids)
        override = self.decode_overrides.get(token_ids)
        if override is not None:
            return override
        return bytes(token_ids).decode("utf-8")


@dataclass(frozen=True, slots=True)
class GenerationScript:
    content_token_ids: tuple[int, ...]
    stop_token_ids: tuple[int, ...] = ()
    finish_reason: str = "stop"
    policy_snapshot_id: str | None = None
    backend_id: str | None = None
    error: Exception | None = None

    @classmethod
    def text(
        cls,
        tokenizer: ByteTokenizer,
        text: str,
        *,
        stop_token_ids: tuple[int, ...] = (900,),
        policy_snapshot_id: str | None = None,
        backend_id: str | None = None,
    ) -> GenerationScript:
        return cls(
            content_token_ids=tuple(tokenizer.encode(text)),
            stop_token_ids=stop_token_ids,
            policy_snapshot_id=policy_snapshot_id,
            backend_id=backend_id,
        )


@dataclass(slots=True)
class ScriptedRolloutGenerator:
    tokenizer: ByteTokenizer
    policy_snapshot: PolicySnapshot
    scripts: list[GenerationScript]
    snapshot_sequence: tuple[PolicySnapshot, ...] = ()
    requests: list[RolloutGenerationRequest] = field(default_factory=list)
    begun_episodes: list[str] = field(default_factory=list)
    ended_episodes: list[str] = field(default_factory=list)
    snapshot_calls: int = 0

    def snapshot(self) -> PolicySnapshot:
        if self.snapshot_sequence:
            index = min(self.snapshot_calls, len(self.snapshot_sequence) - 1)
            result = self.snapshot_sequence[index]
        else:
            result = self.policy_snapshot
        self.snapshot_calls += 1
        return result

    async def generate(
        self,
        request: RolloutGenerationRequest,
    ) -> RolloutGenerationResult:
        self.requests.append(request)
        if not self.scripts:
            raise RuntimeError("scripted generator has no result for this call")
        script = self.scripts.pop(0)
        if script.error is not None:
            raise script.error
        return RolloutGenerationResult(
            content_token_ids=script.content_token_ids,
            stop_token_ids=script.stop_token_ids,
            finish_reason=script.finish_reason,
            policy_snapshot_id=(
                self.policy_snapshot.snapshot_id
                if script.policy_snapshot_id is None
                else script.policy_snapshot_id
            ),
            backend_id=(
                self.policy_snapshot.backend_id if script.backend_id is None else script.backend_id
            ),
            usage=BudgetVector(
                input_tokens=len(request.input_ids),
                output_tokens=len(script.content_token_ids) + len(script.stop_token_ids),
                model_calls=1,
            ),
        )

    def begin_episode(self, episode_id: str, expected_policy_snapshot_id: str) -> None:
        del expected_policy_snapshot_id
        self.begun_episodes.append(episode_id)

    def end_episode(self, episode_id: str) -> None:
        self.ended_episodes.append(episode_id)


@dataclass(slots=True)
class ScriptedEnvironment:
    results: list[EnvironmentObservation | Exception]
    environment_id: str = "debug-environment"
    task_family: str = "debug-family"
    calls: list[tuple[StructuredAction, int]] = field(default_factory=list)
    completion_checks: list[JsonValue] = field(default_factory=list)
    cursor: int = 0

    async def execute(
        self,
        action: StructuredAction,
        *,
        step_index: int,
    ) -> EnvironmentObservation:
        self.calls.append((action, step_index))
        if not self.results:
            raise RuntimeError("scripted environment has no execution result")
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        self.cursor += 1
        return result

    def validate_completion(self, submission: JsonValue) -> bool:
        normalized = normalize_json(submission)
        self.completion_checks.append(normalized)
        return isinstance(normalized, dict) and set(normalized) == {"answer"}


@dataclass(slots=True)
class FakeTerminalEvaluator:
    value: float
    environment_id: str = "debug-environment"
    private_canary: str = PRIVATE_CANARY
    fail: bool = False
    requests: list[TerminalEvaluationRequest] = field(default_factory=list)

    async def evaluate(self, request: TerminalEvaluationRequest) -> TerminalReward:
        self.requests.append(request)
        if self.fail:
            raise TerminalEvaluatorError("private evaluator infrastructure detail")
        return TerminalReward(
            value=self.value,
            success=self.value >= 0.5,
            success_rule=SuccessRule.R_AT_THRESHOLD,
            success_threshold=0.5,
            native_metric_name="debug-score",
            native_payload={"private_diagnostic": self.private_canary},
            environment_id=self.environment_id,
            verifier_version="debug-verifier-v1",
        )


class FakeScoringBackbone:
    """Minimal tensor-returning backbone proving external Phase 3 consumption."""

    def __init__(self, tokenizer: ByteTokenizer) -> None:
        self.tokenizer = tokenizer
        self._forward = torch.tensor(-0.4, requires_grad=True)
        self._backward = torch.tensor(-0.6, requires_grad=True)
        self._z = torch.tensor(0.2, requires_grad=True)

    def score(
        self,
        prefix_ids: tuple[int, ...],
        action_ids: tuple[int, ...],
        role: AdapterRole,
    ) -> torch.Tensor:
        del prefix_ids
        base = self._forward if role is AdapterRole.FORWARD_POLICY else self._backward
        return base + torch.zeros(len(action_ids), dtype=torch.float32)

    def z_value(self, context_ids: tuple[int, ...]) -> torch.Tensor:
        del context_ids
        return self._z


@dataclass(slots=True)
class EngineHarness:
    engine: RolloutEngine
    request: RolloutRequest
    tokenizer: ByteTokenizer
    generator: ScriptedRolloutGenerator
    environment: ScriptedEnvironment
    evaluator: FakeTerminalEvaluator
    ledger: BudgetLedger
    emitter: RuntimeEventEmitter


def default_snapshot(tokenizer: ByteTokenizer) -> PolicySnapshot:
    return PolicySnapshot.create(
        backbone_id="debug-backbone",
        forward_adapter_version="forward@1",
        tokenizer_id=tokenizer.tokenizer_id,
        backend_id="scripted-local",
        initial_trainable_state_hash="initial-trainable-state-debug-v1",
    )


def default_request(
    *,
    trajectory_id: str = "trajectory-one",
    environment_id: str = "debug-environment",
    task_family: str = "debug-family",
) -> RolloutRequest:
    skill_id = "model-selected-skill"
    retrieved = (
        FullRetrievedSkillContext(
            metadata=SkillMetadata(
                skill_id=skill_id,
                version="v1",
                content_hash=stable_hash({"skill": skill_id}),
                input_schema_id="debug-input@1",
                output_schema_id="debug-output@1",
                license_id="unit-test",
                provenance_hash=stable_hash({"source": "engine-fake", "skill": skill_id}),
            ),
            content="Use the model-selected public debug skill.",
            inclusion_reason=RetrievalInclusionReason.APPLICABILITY_MATCH,
        ),
    )
    task = RolloutTask(
        task_id="task-one",
        environment_id=environment_id,
        task_family=task_family,
        context_id=task_family,
        query="Return a public debug answer.",
        available_tools=(),
        public_context={"mode": "public-debug"},
    )
    return RolloutRequest(
        trajectory_id=trajectory_id,
        task=task,
        retrieved_skills=retrieved,
        active_skill_ids=(skill_id,),
        library_version="library-v1",
        sampling_coordinate=ScientificSamplingCoordinate(
            sampling_schedule_hash=stable_hash({"sampling": 17}),
            schedule_purpose="unit-test",
            ordered_sequence_hash=stable_hash([task.task_id]),
            sequence_position=0,
            task_id=task.task_id,
            optimizer_step_or_anchor_ordinal=0,
        ),
        decoding=DecodingSnapshot.create(
            max_reasoning_tokens=128,
            max_action_tokens=4096,
            base_seed=17,
        ),
        epsilon_min=0.05,
        condition_id="trained-skillev@1",
        initial_context_profile=InitialContextProfile.TRAINED_SKILLEV,
    )


def make_harness(
    tmp_path: Path,
    *,
    scripts: list[GenerationScript],
    environment_results: list[EnvironmentObservation | Exception] | None = None,
    tokenizer: ByteTokenizer | None = None,
    snapshot: PolicySnapshot | None = None,
    snapshot_sequence: tuple[PolicySnapshot, ...] = (),
    environment: ScriptedEnvironment | None = None,
    evaluator: FakeTerminalEvaluator | None = None,
    request: RolloutRequest | None = None,
    context_assembler: CanonicalInitialContextAssembler | None = None,
    action_codec: StructuredJsonActionCodec | None = None,
    max_turns: int | None = None,
    event_name: str = "events.jsonl",
) -> EngineHarness:
    tokenizer = tokenizer or ByteTokenizer()
    snapshot = snapshot or default_snapshot(tokenizer)
    generator = ScriptedRolloutGenerator(
        tokenizer=tokenizer,
        policy_snapshot=snapshot,
        scripts=list(scripts),
        snapshot_sequence=snapshot_sequence,
    )
    environment = environment or ScriptedEnvironment(list(environment_results or []))
    evaluator = evaluator or FakeTerminalEvaluator(value=0.75)
    request = request or default_request()
    context_assembler = context_assembler or CanonicalInitialContextAssembler(
        maximum_h0_tokens=4096
    )
    action_codec = action_codec or StructuredJsonActionCodec()
    max_turns = max_turns or max(1, len(scripts) // 2)
    ledger = BudgetLedger(
        run_id="rollout-run",
        attempt_id="rollout-attempt",
        cap=BudgetVector(
            input_tokens=2_000_000,
            output_tokens=100_000,
            model_calls=32,
            agent_turns=16,
            tool_calls=16,
            wall_time_milliseconds=10_000,
        ),
    )
    emitter = RuntimeEventEmitter(
        LiveAttemptEventLog(
            tmp_path / event_name,
            run_id="rollout-run",
            attempt_id="rollout-attempt",
        ),
        producer_id="rollout-engine",
        clock=lambda: FIXED_TIME,
    )
    bounded_agent = BoundedAgent(
        environment=environment,
        policy=BoundedAgentPolicy(max_turns=max_turns),
        ledger=ledger,
        tool_call_maximum=BudgetVector(
            tool_calls=1,
            wall_time_milliseconds=100,
        ),
        emitter=emitter,
    )
    engine = RolloutEngine(
        generator=generator,
        context_assembler=context_assembler,
        action_codec=action_codec,
        bounded_agent=bounded_agent,
        terminal_evaluator=evaluator,
        ledger=ledger,
        reasoning_call_maximum=BudgetVector(
            input_tokens=50_000,
            output_tokens=request.decoding.max_reasoning_tokens,
            model_calls=1,
        ),
        action_call_maximum=BudgetVector(
            input_tokens=50_000,
            output_tokens=request.decoding.max_action_tokens,
            model_calls=1,
            agent_turns=1,
        ),
        emitter=emitter,
        clock=lambda: FIXED_TIME,
    )
    return EngineHarness(
        engine=engine,
        request=request,
        tokenizer=tokenizer,
        generator=generator,
        environment=environment,
        evaluator=evaluator,
        ledger=ledger,
        emitter=emitter,
    )
