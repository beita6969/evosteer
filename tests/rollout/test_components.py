from __future__ import annotations

import copy
import random
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest

from skillev.contracts import (
    ScientificSamplingCoordinate,
    SuccessRule,
    TerminalReward,
    canonical_json,
    normalize_json,
    stable_hash,
)
from skillev.rollout import (
    MAXIMUM_APPLICABLE_SKILL_POSITION,
    ActionDraft,
    ActionParseStatus,
    AssembledInitialContext,
    CanonicalInitialContextAssembler,
    CompletedStepDraft,
    DecodingSnapshot,
    GenerationPhase,
    PolicySnapshot,
    ReasoningDraft,
    RolloutArtifact,
    RolloutBoundaryError,
    RolloutGenerationResult,
    RolloutInfrastructureFailure,
    RolloutInfrastructureKind,
    RolloutManifest,
    RolloutRequest,
    RolloutTask,
    RolloutTermination,
    StructuredJsonActionCodec,
    decode_action_segment,
    decode_reasoning_segment,
    derive_generation_seed,
    finalize_trajectory_record,
    materialize_trajectory_step,
    maximum_retrieved_skill_block_token_count,
    render_retrieved_skill_block,
    retrieved_skill_block_token_count,
)
from skillev.rollout.prompt_profiles import InitialContextProfile
from skillev.runtime import (
    ActionKind,
    BudgetVector,
    EnvironmentObservation,
    FullRetrievedSkillContext,
    RetrievalInclusionReason,
    SkillMetadata,
)
from skillev.scoring import (
    render_forward_prefix_from_parts,
    render_hindsight_prefix_from_parts,
    render_reasoning_prefix,
)


class CharacterTokenizer:
    """Small exact tokenizer whose IDs preserve every generated character."""

    @property
    def tokenizer_id(self) -> str:
        return "character-tokenizer@1"

    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


class ManyToOneTokenizer(CharacterTokenizer):
    sampled_ids = (10, 11)
    canonical_ids = (999,)
    action_text = "decoded-but-different"

    @property
    def tokenizer_id(self) -> str:
        return "many-to-one-tokenizer@1"

    def encode(self, text: str) -> list[int]:
        if text == self.action_text:
            return list(self.canonical_ids)
        return super().encode(text)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        if token_ids in {self.sampled_ids, self.canonical_ids}:
            return self.action_text
        return super().decode(token_ids)


class EmptyDecodeTokenizer(CharacterTokenizer):
    def decode(self, token_ids: tuple[int, ...]) -> str:
        del token_ids
        return ""


def _skill(skill_id: str, *, content: str) -> FullRetrievedSkillContext:
    return FullRetrievedSkillContext(
        metadata=SkillMetadata(
            skill_id=skill_id,
            version="v1",
            content_hash=stable_hash({"content": content, "skill_id": skill_id}),
            input_schema_id="schema-input-v1",
            output_schema_id="schema-output-v1",
            license_id="internal-test",
            provenance_hash=stable_hash({"origin": "phase-four-cpu-test", "skill_id": skill_id}),
        ),
        content=content,
        inclusion_reason=RetrievalInclusionReason.APPLICABILITY_MATCH,
    )


def _task() -> RolloutTask:
    return RolloutTask(
        task_id="task-public-1",
        environment_id="environment-public",
        task_family="cpu-debug",
        context_id="cpu-debug",
        query="Solve the public debug task.",
        available_tools=(),
        public_context={"zeta": 2, "alpha": {"visible": True}},
    )


def _benchmark_task(benchmark_id: str, tools: tuple[str, ...]) -> RolloutTask:
    return RolloutTask(
        task_id=f"{benchmark_id}/public-task-1",
        environment_id=f"benchmark:{benchmark_id}@test",
        task_family=f"{benchmark_id}/test",
        context_id=f"{benchmark_id}:test",
        query="Solve this answer-free public benchmark task.",
        available_tools=tools,
        public_context={"benchmark_id": benchmark_id, "public": True},
    )


def _decoding() -> DecodingSnapshot:
    return DecodingSnapshot.create(
        max_reasoning_tokens=16,
        max_action_tokens=64,
        base_seed=20260720,
    )


def _snapshot(*, tokenizer_id: str = "character-tokenizer@1") -> PolicySnapshot:
    return PolicySnapshot.create(
        backbone_id="tiny-backbone-v1",
        forward_adapter_version="forward-adapter-v3",
        tokenizer_id=tokenizer_id,
        backend_id="scripted-local",
        initial_trainable_state_hash="initial-trainable-state-components-v1",
    )


def _assemble(
    tokenizer: CharacterTokenizer,
) -> tuple[RolloutTask, tuple[FullRetrievedSkillContext, ...], AssembledInitialContext]:
    task = _task()
    skills = (
        _skill("skill-zeta", content="Use the zeta procedure."),
        _skill("skill-alpha", content="Use the alpha procedure."),
    )
    assembled = CanonicalInitialContextAssembler(maximum_h0_tokens=4096).assemble(
        task=task,
        retrieved_skills=skills,
        active_skill_ids=tuple(sorted(skill.metadata.skill_id for skill in skills)),
        library_version="library-v5",
        tokenizer=tokenizer,
    )
    return task, skills, assembled


def _request(
    *,
    task: RolloutTask,
    skills: tuple[FullRetrievedSkillContext, ...],
) -> RolloutRequest:
    return RolloutRequest(
        trajectory_id="trajectory-rollout-components",
        task=task,
        retrieved_skills=skills,
        active_skill_ids=tuple(sorted(skill.metadata.skill_id for skill in skills)),
        library_version="library-v5",
        sampling_coordinate=ScientificSamplingCoordinate(
            sampling_schedule_hash=stable_hash({"sampling": 17}),
            schedule_purpose="unit-test",
            ordered_sequence_hash=stable_hash([task.task_id]),
            sequence_position=0,
            task_id=task.task_id,
            optimizer_step_or_anchor_ordinal=0,
        ),
        decoding=_decoding(),
        epsilon_min=0.01,
        condition_id="trained-skillev@1",
        initial_context_profile=InitialContextProfile.TRAINED_SKILLEV,
    )


def _valid_action_text() -> str:
    return (
        '{"kind": "skill", "name": "lookup", '
        '"arguments": {"query": "public"}, '
        '"resource_id": "debug.skill", "skill_id": "skill-zeta"}'
    )


def _materialized_step(
    *,
    tokenizer: CharacterTokenizer,
    assembled: AssembledInitialContext,
    snapshot: PolicySnapshot,
) -> tuple[ReasoningDraft, ActionDraft, EnvironmentObservation, Any]:
    reasoning_text = "Inspect the public evidence."
    reasoning_prompt = render_reasoning_prefix(assembled.text, (), 1)
    forward = render_forward_prefix_from_parts(assembled.text, (), 1, reasoning_text)
    action_text = _valid_action_text()
    action_token_ids = tuple(tokenizer.encode(action_text))
    parse_result = StructuredJsonActionCodec().parse(action_text)
    observation = EnvironmentObservation(
        public_value={"result": "public-ok"},
        observation_status="success",
        invoked_skill_ids=("skill-zeta",),
    )
    reasoning = ReasoningDraft(
        step_index=1,
        prompt_text=reasoning_prompt.text,
        prompt_hash=reasoning_prompt.prompt_hash,
        text=reasoning_text,
        generated_token_ids=tuple(tokenizer.encode(reasoning_text)),
        policy_snapshot_id=snapshot.snapshot_id,
    )
    action = ActionDraft(
        step_index=1,
        forward_prefix_text=forward.text,
        forward_prefix_hash=forward.prefix_hash,
        text=action_text,
        token_ids=action_token_ids,
        parse_result=parse_result,
        policy_snapshot_id=snapshot.snapshot_id,
    )
    step = materialize_trajectory_step(
        initial_text=assembled.text,
        previous_steps=(),
        draft=CompletedStepDraft(
            reasoning=reasoning,
            action=action,
            observation=observation,
        ),
    )
    return reasoning, action, observation, step


def _artifact() -> tuple[CharacterTokenizer, RolloutArtifact]:
    tokenizer = CharacterTokenizer()
    task, skills, assembled = _assemble(tokenizer)
    request = _request(task=task, skills=skills)
    snapshot = _snapshot()
    reasoning, _, _, step = _materialized_step(
        tokenizer=tokenizer,
        assembled=assembled,
        snapshot=snapshot,
    )
    reward = TerminalReward(
        value=0.75,
        success=True,
        success_rule=SuccessRule.R_AT_THRESHOLD,
        success_threshold=0.5,
        native_metric_name="cpu-debug-score",
        native_payload={"public_metric": 0.75},
        environment_id=task.environment_id,
        verifier_version="fake-private-evaluator-v1",
    )
    record = finalize_trajectory_record(
        request=request,
        assembled=assembled,
        steps=(step,),
        reward=reward,
        tokenizer=tokenizer,
        created_at="2026-07-20T00:00:01Z",
    )
    manifest = RolloutManifest(
        trajectory_id=request.trajectory_id,
        task_id=task.task_id,
        policy_snapshot=snapshot,
        library_version=request.library_version,
        sampling_coordinate=request.sampling_coordinate,
        decoding_snapshot_id=request.decoding.snapshot_id,
        assembler_version=assembled.contract.assembler_version,
        action_format_version=StructuredJsonActionCodec.format_version,
        generator_backend_id=snapshot.backend_id,
        termination=RolloutTermination.HORIZON_EXHAUSTED,
        reasoning_token_counts=(len(reasoning.generated_token_ids),),
        started_at="2026-07-20T00:00:00Z",
        completed_at="2026-07-20T00:00:02Z",
    )
    artifact = RolloutArtifact(
        initial_context=assembled,
        record=record,
        manifest=manifest,
    )
    return tokenizer, artifact


def test_policy_and_decoding_snapshots_bind_all_identity_fields() -> None:
    policy = _snapshot()
    decoding = _decoding()

    assert PolicySnapshot.from_value(policy.to_value()) == policy
    assert DecodingSnapshot.from_value(decoding.to_value()) == decoding
    assert normalize_json(policy.to_value()) == policy.to_value()
    assert normalize_json(decoding.to_value()) == decoding.to_value()

    with pytest.raises(ValueError):
        replace(policy, forward_adapter_version="forward-adapter-stale")
    with pytest.raises(ValueError):
        replace(decoding, max_action_tokens=decoding.max_action_tokens + 1)
    with pytest.raises(ValueError):
        DecodingSnapshot.create(
            max_reasoning_tokens=16,
            max_action_tokens=64,
            base_seed=2**64,
        )


def test_seed_derivation_is_stable_distinct_and_does_not_touch_global_rng() -> None:
    random.seed(314159)
    rng_state = random.getstate()
    coordinate = ScientificSamplingCoordinate(
        sampling_schedule_hash=stable_hash({"sampling": 17}),
        schedule_purpose="iid-training",
        ordered_sequence_hash=stable_hash(["task-one"]),
        sequence_position=0,
        task_id="task-one",
        optimizer_step_or_anchor_ordinal=1,
    )

    reasoning_one = derive_generation_seed(
        base_seed=17,
        coordinate=coordinate,
        step_index=1,
        phase=GenerationPhase.REASONING,
    )
    reasoning_one_again = derive_generation_seed(
        base_seed=17,
        coordinate=coordinate,
        step_index=1,
        phase=GenerationPhase.REASONING,
    )
    action_one = derive_generation_seed(
        base_seed=17,
        coordinate=coordinate,
        step_index=1,
        phase=GenerationPhase.ACTION,
    )
    reasoning_two = derive_generation_seed(
        base_seed=17,
        coordinate=coordinate,
        step_index=2,
        phase=GenerationPhase.REASONING,
    )

    assert reasoning_one == reasoning_one_again
    assert len({reasoning_one, action_one, reasoning_two}) == 3
    assert all(0 <= seed < 2**64 for seed in (reasoning_one, action_one, reasoning_two))
    assert random.getstate() == rng_state


def test_scientific_sampling_coordinate_is_artifact_namespace_free_and_round_trips() -> None:
    coordinate = ScientificSamplingCoordinate(
        sampling_schedule_hash=stable_hash({"sampling": 17}),
        schedule_purpose="iid-evaluation",
        ordered_sequence_hash=stable_hash(["task-one"]),
        sequence_position=0,
        task_id="task-one",
        optimizer_step_or_anchor_ordinal=0,
    )

    assert ScientificSamplingCoordinate.from_value(coordinate.to_value()) == coordinate
    assert set(coordinate.to_value()).isdisjoint(
        {
            "experiment_id",
            "run_id",
            "attempt_id",
            "output_directory",
            "source_arm_identity",
            "state_hash",
            "trajectory_id",
        }
    )
    seed = derive_generation_seed(
        base_seed=17,
        coordinate=coordinate,
        step_index=1,
        phase=GenerationPhase.ACTION,
    )
    assert seed == derive_generation_seed(
        base_seed=17,
        coordinate=ScientificSamplingCoordinate.from_value(coordinate.to_value()),
        step_index=1,
        phase=GenerationPhase.ACTION,
    )


def test_initial_context_preserves_skill_order_and_uses_canonical_public_json() -> None:
    tokenizer = CharacterTokenizer()
    task, skills, assembled = _assemble(tokenizer)

    assert assembled.contract.retrieved_skill_ids == ("skill-zeta", "skill-alpha")
    assert assembled.contract.meta["retrieval_inclusions"] == [
        {"inclusion_reason": "applicability-match", "skill_id": "skill-zeta"},
        {"inclusion_reason": "applicability-match", "skill_id": "skill-alpha"},
    ]
    assert assembled.text.index("[1] id=skill-zeta") < assembled.text.index("[2] id=skill-alpha")
    assert canonical_json(task.public_context) in assembled.text
    assert "{'zeta':" not in assembled.text
    assert assembled.contract.assembled_token_count == len(tokenizer.encode(assembled.text))
    assert AssembledInitialContext.from_value(assembled.to_value()) == assembled

    request = _request(task=task, skills=skills)
    assert request.retrieved_skills == skills
    assert RolloutTask.from_value(task.to_value()) == task


def test_rollout_task_normalizes_unicode_before_wire_round_trip() -> None:
    task = RolloutTask.from_value(
        {
            "available_tools": [],
            "context_id": "context",
            "environment_id": "environment",
            "public_context": {},
            "query": "Cafe\u0301",
            "task_family": "fixture/completion",
            "task_id": "task",
        }
    )

    assert task.query == "Caf\u00e9"
    assert RolloutTask.from_value(task.to_value()) == task


def test_retrieved_skill_block_renderer_is_the_exact_h0_counting_authority() -> None:
    tokenizer = CharacterTokenizer()
    task, skills, assembled = _assemble(tokenizer)
    del task
    first = skills[0]
    metadata = first.metadata
    expected = (
        f"[1] id={metadata.skill_id}\n"
        f"version={metadata.version}\n"
        f"content_hash={metadata.content_hash}\n"
        f"content:\n{first.content}\n"
    )

    assert render_retrieved_skill_block(first, position=1) == expected
    assert retrieved_skill_block_token_count(
        first,
        position=1,
        tokenizer=tokenizer,
    ) == len(tokenizer.encode(expected))
    assert maximum_retrieved_skill_block_token_count(
        first,
        maximum_position=MAXIMUM_APPLICABLE_SKILL_POSITION,
        tokenizer=tokenizer,
    ) == max(
        len(tokenizer.encode(render_retrieved_skill_block(first, position=position)))
        for position in range(1, MAXIMUM_APPLICABLE_SKILL_POSITION + 1)
    )
    assert expected in assembled.text


@pytest.mark.parametrize(
    ("benchmark_id", "tools", "expected_actions"),
    [
        (
            "hotpotqa",
            (),
            ((ActionKind.COMPLETE, None, "complete"),),
        ),
        (
            "webshop",
            ("click", "purchase", "search"),
            (
                (ActionKind.TOOL, "webshop", "search"),
                (ActionKind.TOOL, "webshop", "click"),
                (ActionKind.TOOL, "webshop", "purchase"),
            ),
        ),
        (
            "alfworld",
            ("act",),
            ((ActionKind.TOOL, "alfworld", "act"),),
        ),
        (
            "spreadsheetbench",
            ("spreadsheet.execute", "submit"),
            (
                (ActionKind.TOOL, "spreadsheet", "execute"),
                (ActionKind.COMPLETE, None, "complete"),
            ),
        ),
        (
            "appworld",
            ("appworld.execute", "submit"),
            (
                (ActionKind.TOOL, "appworld", "execute"),
                (ActionKind.COMPLETE, None, "complete"),
            ),
        ),
    ],
)
def test_initial_context_exposes_parseable_benchmark_action_contracts(
    benchmark_id: str,
    tools: tuple[str, ...],
    expected_actions: tuple[tuple[ActionKind, str | None, str], ...],
) -> None:
    task = _benchmark_task(benchmark_id, tools)
    assembled = CanonicalInitialContextAssembler(maximum_h0_tokens=20_000).assemble(
        task=task,
        retrieved_skills=(),
        active_skill_ids=(),
        library_version="library-v5",
        tokenizer=CharacterTokenizer(),
    )
    codec = StructuredJsonActionCodec()
    parsed = [
        codec.parse(line).action
        for line in assembled.text.splitlines()
        if line.startswith('{"arguments":')
    ]

    assert codec.format_version in assembled.text
    assert all(action is not None for action in parsed)
    assert (
        tuple(
            (action.kind, action.resource_id, action.name)
            for action in parsed
            if action is not None
        )
        == expected_actions
    )
    if benchmark_id == "spreadsheetbench":
        assert parsed[-1] is not None
        assert parsed[-1].arguments == {"value": {"submit": True}}
    if benchmark_id == "appworld":
        assert parsed[-1] is not None
        assert parsed[-1].arguments == {"value": {"submit": True}}


def test_initial_context_and_request_reject_duplicate_skill_ids() -> None:
    tokenizer = CharacterTokenizer()
    task = _task()
    duplicate = _skill("skill-duplicate", content="Visible instructions.")
    skills = (duplicate, duplicate)

    with pytest.raises(ValueError):
        CanonicalInitialContextAssembler(maximum_h0_tokens=4096).assemble(
            task=task,
            retrieved_skills=skills,
            active_skill_ids=("skill-duplicate",),
            library_version="library-v5",
            tokenizer=tokenizer,
        )
    with pytest.raises(ValueError):
        _request(task=task, skills=skills)


def test_action_codec_separates_json_parse_and_schema_failures() -> None:
    codec = StructuredJsonActionCodec()

    valid = codec.parse(_valid_action_text())
    parse_error = codec.parse("{not-json")
    schema_invalid = codec.parse('{"kind":"tool","name":"lookup","arguments":{}}')
    invalid_skill = codec.parse(
        '{"arguments":{},"kind":"skill","name":"apply-skill",'
        '"resource_id":"skill-runtime","skill_id":"None"}'
    )

    assert valid.status is ActionParseStatus.VALID
    assert valid.action is not None
    assert valid.action.kind is ActionKind.SKILL
    assert valid.action.arguments == {"query": "public"}
    assert parse_error.status is ActionParseStatus.PARSE_ERROR
    assert parse_error.action is None
    assert schema_invalid.status is ActionParseStatus.SCHEMA_INVALID
    assert schema_invalid.action is None
    assert invalid_skill.status is ActionParseStatus.SCHEMA_INVALID
    assert invalid_skill.action is None


def test_action_decode_accepts_many_to_one_span_and_preserves_original_ids() -> None:
    tokenizer = ManyToOneTokenizer()
    generated_ids = tokenizer.sampled_ids

    decoded = decode_action_segment(tokenizer, generated_ids)

    assert decoded.text == tokenizer.action_text
    assert decoded.token_ids is generated_ids


def test_action_decode_rejects_empty_model_visible_text() -> None:
    with pytest.raises(RolloutBoundaryError):
        decode_action_segment(EmptyDecodeTokenizer(), (10, 11))


def test_reasoning_decode_allows_an_empty_generated_segment() -> None:
    decoded = decode_reasoning_segment(CharacterTokenizer(), ())

    assert decoded.text == ""
    assert decoded.token_ids == ()


def test_step_materialization_preserves_action_tokens_and_canonical_commitments() -> None:
    tokenizer = CharacterTokenizer()
    _, _, assembled = _assemble(tokenizer)
    snapshot = _snapshot()
    reasoning, action, observation, step = _materialized_step(
        tokenizer=tokenizer,
        assembled=assembled,
        snapshot=snapshot,
    )

    assert step.reasoning_text == reasoning.text
    assert step.action_text == action.text
    assert step.action_token_ids is action.token_ids
    assert step.action_token_count == len(action.token_ids)
    assert step.observation_text == canonical_json(observation.public_value)
    assert step.observation_status == observation.observation_status
    assert step.invoked_skill_ids == observation.invoked_skill_ids
    assert step.forward_prefix_hash == action.forward_prefix_hash
    assert (
        step.hindsight_prefix_hash
        == render_hindsight_prefix_from_parts(
            assembled.text,
            (),
            1,
            observation.observation_text,
        ).prefix_hash
    )

    with pytest.raises(ValueError):
        materialize_trajectory_step(
            initial_text=assembled.text,
            previous_steps=(),
            draft=CompletedStepDraft(
                reasoning=reasoning,
                action=replace(action, forward_prefix_text=action.forward_prefix_text + "drift"),
                observation=observation,
            ),
        )


def test_invalid_action_is_still_materialized_with_original_span() -> None:
    tokenizer = CharacterTokenizer()
    _, _, assembled = _assemble(tokenizer)
    snapshot = _snapshot()
    reasoning_prompt = render_reasoning_prefix(assembled.text, (), 1)
    reasoning_text = ""
    forward = render_forward_prefix_from_parts(assembled.text, (), 1, reasoning_text)
    action_text = "{not-json"
    action_ids = tuple(tokenizer.encode(action_text))
    parse_result = StructuredJsonActionCodec().parse(action_text)
    observation = EnvironmentObservation(
        public_value={"error": "action_not_json"},
        observation_status="parse_error",
    )
    draft = CompletedStepDraft(
        reasoning=ReasoningDraft(
            step_index=1,
            prompt_text=reasoning_prompt.text,
            prompt_hash=reasoning_prompt.prompt_hash,
            text=reasoning_text,
            generated_token_ids=(),
            policy_snapshot_id=snapshot.snapshot_id,
        ),
        action=ActionDraft(
            step_index=1,
            forward_prefix_text=forward.text,
            forward_prefix_hash=forward.prefix_hash,
            text=action_text,
            token_ids=action_ids,
            parse_result=parse_result,
            policy_snapshot_id=snapshot.snapshot_id,
        ),
        observation=observation,
    )

    step = materialize_trajectory_step(
        initial_text=assembled.text,
        previous_steps=(),
        draft=draft,
    )

    assert step.action_text == action_text
    assert step.action_token_ids is action_ids
    assert step.observation_status == "parse_error"


def test_manifest_and_artifact_have_canonical_admitted_round_trips() -> None:
    tokenizer, artifact = _artifact()

    loaded_manifest = RolloutManifest.from_value(artifact.manifest.to_value())
    loaded_artifact = RolloutArtifact.from_value(artifact.to_value(), tokenizer=tokenizer)

    assert loaded_manifest == artifact.manifest
    assert loaded_artifact == artifact
    for value in (artifact.manifest.to_value(), artifact.to_value()):
        assert normalize_json(value) == value
    assert canonical_json(loaded_artifact.to_value()) == canonical_json(artifact.to_value())


def test_artifact_reload_preserves_many_to_one_sampled_action_span() -> None:
    base_tokenizer, artifact = _artifact()
    action_text = artifact.record.steps[0].action_text
    sampled_ids = (41, 42)

    class ArtifactManyToOneTokenizer(CharacterTokenizer):
        def decode(self, token_ids: tuple[int, ...]) -> str:
            if token_ids == sampled_ids:
                return action_text
            return super().decode(token_ids)

    tokenizer = ArtifactManyToOneTokenizer()
    serialized: Any = copy.deepcopy(artifact.to_value())
    serialized["record"]["steps"][0]["action_token_ids"] = list(sampled_ids)
    serialized["record"]["steps"][0]["action_token_count"] = len(sampled_ids)

    restored = RolloutArtifact.from_value(serialized, tokenizer=tokenizer)

    assert restored.record.steps[0].action_token_ids == sampled_ids
    assert tuple(base_tokenizer.encode(action_text)) != sampled_ids

    tampered_text = copy.deepcopy(serialized)
    tampered_text["record"]["steps"][0]["action_text"] = "tampered"
    with pytest.raises(ValueError):
        RolloutArtifact.from_value(tampered_text, tokenizer=tokenizer)

    tampered_ids = copy.deepcopy(serialized)
    tampered_ids["record"]["steps"][0]["action_token_ids"] = [43]
    tampered_ids["record"]["steps"][0]["action_token_count"] = 1
    with pytest.raises(ValueError):
        RolloutArtifact.from_value(tampered_ids, tokenizer=tokenizer)


def _assert_strict_wire_admission(
    loader: Callable[[object], object],
    value: dict[str, object],
    *,
    missing_key: str,
    wrong_key: str,
    wrong_value: object,
) -> None:
    loader(copy.deepcopy(value))

    missing = copy.deepcopy(value)
    del missing[missing_key]
    extra = copy.deepcopy(value)
    extra["unexpected"] = "not-part-of-the-wire-schema"
    wrong_type = copy.deepcopy(value)
    wrong_type[wrong_key] = wrong_value

    for mutation in (missing, extra, wrong_type):
        with pytest.raises((TypeError, ValueError)):
            loader(mutation)


def test_rollout_wire_loaders_require_exact_fields_and_wire_types() -> None:
    tokenizer, artifact = _artifact()
    task, skills, assembled = _assemble(tokenizer)
    request = _request(task=task, skills=skills)
    snapshot = _snapshot()
    decoding = _decoding()
    generation = RolloutGenerationResult(
        content_token_ids=(101, 102),
        stop_token_ids=(103,),
        finish_reason="stop",
        policy_snapshot_id=snapshot.snapshot_id,
        backend_id=snapshot.backend_id,
        usage=BudgetVector(output_tokens=3, model_calls=1),
    )
    failure = RolloutInfrastructureFailure(
        trajectory_id=request.trajectory_id,
        task_id=request.task.task_id,
        kind=RolloutInfrastructureKind.INITIAL_CONTEXT_MISMATCH,
        stage="restore",
        step_index=None,
        public_message="rollout state is unavailable",
    )

    cases = (
        (PolicySnapshot.from_value, snapshot.to_value(), "backend_id", "backbone_id", 1),
        (
            DecodingSnapshot.from_value,
            decoding.to_value(),
            "format_version",
            "max_action_tokens",
            "64",
        ),
        (RolloutTask.from_value, task.to_value(), "public_context", "query", 1),
        (
            AssembledInitialContext.from_value,
            assembled.to_value(),
            "text",
            "text",
            1,
        ),
        (
            RolloutGenerationResult.from_value,
            generation.to_value(),
            "backend_id",
            "backend_id",
            7,
        ),
        (
            RolloutInfrastructureFailure.from_value,
            failure.to_value(),
            "step_index",
            "step_index",
            "1",
        ),
        (
            RolloutManifest.from_value,
            artifact.manifest.to_value(),
            "started_at",
            "reasoning_token_counts",
            ["1"],
        ),
        (
            lambda value: RolloutArtifact.from_value(value, tokenizer=tokenizer),
            artifact.to_value(),
            "record",
            "manifest",
            None,
        ),
    )

    for loader, value, missing_key, wrong_key, wrong_value in cases:
        _assert_strict_wire_admission(
            loader,
            value,
            missing_key=missing_key,
            wrong_key=wrong_key,
            wrong_value=wrong_value,
        )


def test_artifact_loader_rejects_tampered_action_token_identity() -> None:
    tokenizer, artifact = _artifact()
    serialized: Any = copy.deepcopy(artifact.to_value())
    recorded_ids = serialized["record"]["steps"][0]["action_token_ids"]
    recorded_ids[0] += 1

    with pytest.raises(ValueError):
        RolloutArtifact.from_value(serialized, tokenizer=tokenizer)


def test_artifact_loader_rechecks_initial_context_token_count() -> None:
    tokenizer, artifact = _artifact()
    serialized: Any = copy.deepcopy(artifact.to_value())
    serialized["initial_context"]["contract"]["assembled_token_count"] += 1
    serialized["record"]["initial_context"]["assembled_token_count"] += 1

    with pytest.raises(ValueError):
        RolloutArtifact.from_value(serialized, tokenizer=tokenizer)


def test_artifact_loader_rejects_manifest_task_identity_tampering() -> None:
    tokenizer, artifact = _artifact()
    serialized: Any = copy.deepcopy(artifact.to_value())
    serialized["manifest"]["task_id"] = "task-routed-elsewhere"

    with pytest.raises(ValueError):
        RolloutArtifact.from_value(serialized, tokenizer=tokenizer)


def test_artifact_loader_rejects_manifest_library_identity_tampering() -> None:
    tokenizer, artifact = _artifact()
    serialized: Any = copy.deepcopy(artifact.to_value())
    serialized["manifest"]["library_version"] = "library-routed-elsewhere"

    with pytest.raises(ValueError):
        RolloutArtifact.from_value(serialized, tokenizer=tokenizer)
