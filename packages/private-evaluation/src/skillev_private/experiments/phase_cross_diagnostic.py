"""Read-only R0/Rk x A0/Ak diagnostic, never an answer-selection workflow."""

from __future__ import annotations

import json

from skillev.contracts import JsonValue, TrajectoryStep
from skillev.contracts.action_wire import NATIVE_TOOL_WIRES
from skillev.policy.interface import ModelInputWindow, encode_policy_prompt
from skillev.policy.phase_context import PhaseContextSpec
from skillev.rollout import (
    GenerationPhase,
    RolloutGenerationRequest,
    RolloutGenerationResult,
    RolloutGenerator,
)
from skillev.rollout.codec import codec_for_initial_meta
from skillev.scoring.rendering import render_forward_prefix_from_parts, render_reasoning_prefix


async def phase_cross(
    *,
    initial: RolloutGenerator,
    trained: RolloutGenerator,
    diagnostic_id: str,
    initial_text: str,
    previous_steps: tuple[TrajectoryStep, ...],
    library_version: str,
    decoding_snapshot_id: str,
    native_thinking: bool,
    reasoning_seed: int,
    action_seed: int,
    max_reasoning_tokens: int,
    max_action_tokens: int,
    input_window: ModelInputWindow | None,
) -> dict[str, JsonValue]:
    """Six requests, fixed public history and budgets; retain all four actions.

    Both reasoning origins are sampled on their own declared policy, rather than
    reusing only trained-policy reasoning and labeling it Step-0. No action is
    executed, scored for task success, selected or converted into training data.
    """
    actors = {"initial": initial, "trained": trained}
    snapshots = {name: actor.snapshot() for name, actor in actors.items()}
    if initial.tokenizer.tokenizer_id != trained.tokenizer.tokenizer_id:
        raise ValueError("phase cross requires the same tokenizer on both policies")
    tokenizer = initial.tokenizer
    spec, _ = PhaseContextSpec.split(initial_text)
    codec = codec_for_initial_meta(
        {}
        if spec is None
        else {
            "action_wire": spec.action_wire,
            "action_contract": json.loads(spec.action_contract_json),
        }
    )
    boundary = (
        "native-model-stop@1"
        if spec is not None and spec.action_wire in NATIVE_TOOL_WIRES
        else "action-json-root@1"
    )
    turn = len(previous_steps) + 1
    reasoning_text = render_reasoning_prefix(initial_text, previous_steps, turn).text
    reasoning_input = encode_policy_prompt(
        tokenizer,
        reasoning_text,
        initial_text=initial_text,
        window=input_window,
        native_thinking=native_thinking,
    )

    async def generate(
        name: str,
        phase: GenerationPhase,
        ids: tuple[int, ...],
        seed: int,
        maximum: int,
        coordinate: str,
    ) -> RolloutGenerationResult:
        actor, snapshot = actors[name], snapshots[name]
        if actor.snapshot() != snapshot:
            raise ValueError("diagnostic policy changed before generation")
        episode = f"{diagnostic_id}:{coordinate}"
        actor.begin_episode(episode, snapshot.snapshot_id)
        try:
            request = RolloutGenerationRequest(
                phase,
                ids,
                maximum,
                seed,
                decoding_snapshot_id,
                snapshot.snapshot_id,
                episode_id=episode,
                turn_index=turn,
                library_version=library_version,
                action_boundary_version=boundary,
            )
            result = await actor.generate(request)
            if result.policy_snapshot_id != snapshot.snapshot_id or actor.snapshot() != snapshot:
                raise ValueError("diagnostic generation used another policy")
            return result
        finally:
            actor.end_episode(episode)

    reasonings: dict[str, JsonValue] = {}
    actions: dict[str, JsonValue] = {}
    for origin in actors:
        reasoning = await generate(
            origin,
            GenerationPhase.REASONING,
            reasoning_input.ids,
            reasoning_seed,
            max_reasoning_tokens,
            f"reasoning-{origin}",
        )
        reasonings[origin] = reasoning.to_value()
        prefix = render_forward_prefix_from_parts(
            initial_text,
            previous_steps,
            turn,
            tokenizer.decode(reasoning.content_token_ids),
        )
        action_input = encode_policy_prompt(
            tokenizer,
            prefix.text,
            initial_text=initial_text,
            window=input_window,
        )
        for destination in actors:
            action = await generate(
                destination,
                GenerationPhase.ACTION,
                action_input.ids,
                action_seed,
                max_action_tokens,
                f"reasoning-{origin}-action-{destination}",
            )
            # Match the production stop-only action rule; never repair the text.
            sampled_ids = action.content_token_ids or action.stop_token_ids
            if not sampled_ids:
                raise ValueError("diagnostic action returned no sampled token")
            parsed = codec.parse(tokenizer.decode(sampled_ids))
            actions[f"R-{origin}/A-{destination}"] = {
                "reasoning_policy_snapshot_id": snapshots[origin].snapshot_id,
                "action_policy_snapshot_id": snapshots[destination].snapshot_id,
                "input_ids": list(action_input.ids),
                "input_tokens_before_window": action_input.original_tokens,
                "generation": action.to_value(),
                "parse_status": parsed.status.value,
                "admission_status": None,
                "terminal_task_success": None,
            }
    return {
        "format": "skillev-phase-cross@1",
        "purpose": "diagnostic-not-task-evaluation",
        "diagnostic_id": diagnostic_id,
        "library_version": library_version,
        "decoding_snapshot_id": decoding_snapshot_id,
        "reasoning_native_thinking": native_thinking,
        "action_native_thinking": False,
        "action_boundary_version": boundary,
        "reasoning_seed": reasoning_seed,
        "action_seed": action_seed,
        "reasoning_input_ids": list(reasoning_input.ids),
        "reasonings": reasonings,
        "actions": actions,
    }
