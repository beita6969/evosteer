from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from typing import Any

from skillev.contracts.canonical import canonical_json, stable_hash
from skillev.contracts.evosteer import DecisionRecord, EvoTask, EvoTrajectory
from skillev.orchestration.actions import GraphAction
from skillev.orchestration.budget import can_reserve
from skillev.orchestration.execution import GraphRuntime
from skillev.orchestration.feature_diagnosis import diagnosis_text
from skillev.runtime.budget_ledger import BudgetLedger
from skillev.runtime.contracts import BudgetReservation, BudgetSettlement, BudgetVector


def actor_projection(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: actor_projection(item)
            for key, item in value.items()
            if key
            not in {
                "runtime_id",
                "executor_identity",
                "node_request",
                "before_state_id",
                "after_state_id",
                "reservation_id",
                "reservation_ids",
                "local_budget_entries",
                "invocation_id",
                "session_id",
                "prompt",
                "tool_transcript",
            }
        }
    if isinstance(value, list):
        return [actor_projection(item) for item in value]
    return value


CONTROLLER_OUTPUT_HEAD_CHARS = 1200
CONTROLLER_OUTPUT_TAIL_CHARS = 1200
LIVE_OUTPUT_REF = "graph.nodes"
HISTORY_OUTPUT_REF = "history"


def elide_output(text: str) -> str:
    if type(text) is not str:
        raise TypeError("node output must be text")
    omitted = len(text) - CONTROLLER_OUTPUT_HEAD_CHARS - CONTROLLER_OUTPUT_TAIL_CHARS
    if omitted <= 0:
        return text
    return (
        text[:CONTROLLER_OUTPUT_HEAD_CHARS]
        + f"[... {omitted} characters omitted ...]"
        + text[-CONTROLLER_OUTPUT_TAIL_CHARS:]
    )


def _version(node_id: Any, output_version: Any) -> tuple[str, int] | None:
    if type(node_id) is not str or type(output_version) is not int:
        return None
    return node_id, output_version


def _reference_messages(
    value: Any, live: dict[tuple[str, int], str], produced: dict[tuple[str, int], str]
) -> None:
    if isinstance(value, list):
        for item in value:
            _reference_messages(item, live, produced)
        return
    if not isinstance(value, dict):
        return
    body = value.get("body")
    key = _version(value.get("source_id"), value.get("source_output_version"))
    if type(body) is str and key is not None:
        del value["body"]
        if live.get(key) == body:
            value["body_ref"] = LIVE_OUTPUT_REF
        elif produced.get(key) == body:
            value["body_ref"] = HISTORY_OUTPUT_REF
        else:
            value["body"] = elide_output(body)
    for item in value.values():
        _reference_messages(item, live, produced)


def controller_projection(state: Any) -> dict[str, Any]:
    projected = actor_projection(state)
    graph = projected.get("graph") if isinstance(projected, dict) else None
    if (
        not isinstance(graph, dict)
        or not isinstance(graph.get("nodes"), list)
        or not isinstance(projected.get("history"), list)
    ):
        raise ValueError("controller projection requires a full public graph state")
    live: dict[tuple[str, int], str] = {}
    for node in graph["nodes"]:
        key = (
            _version(node.get("node_id"), node.get("output_version"))
            if isinstance(node, dict)
            else None
        )
        if key is None or type(node.get("output")) is not str:
            raise ValueError("graph nodes must carry an identified text output")
        live[key] = node["output"]
        node["output"] = elide_output(node["output"])
    produced: dict[tuple[str, int], str] = {}
    observations = [
        event["observation"]
        for event in projected["history"]
        if isinstance(event, dict) and isinstance(event.get("observation"), dict)
    ]
    for observation in observations:
        result = observation.get("node_result")
        if not isinstance(result, dict) or type(result.get("output")) is not str:
            continue
        key = _version(observation.get("node_id"), observation.get("output_version"))
        output = result.pop("output")
        if key is not None and live.get(key) == output:
            result["output_ref"] = LIVE_OUTPUT_REF
        else:
            result["output"] = elide_output(output)
            if key is not None:
                produced.setdefault(key, output)
    for observation in observations:
        for name, item in observation.items():
            if name != "node_result":
                _reference_messages(item, live, produced)
    return projected


async def collect_episode(
    *,
    policy: Any,
    runtime: GraphRuntime,
    ledger: BudgetLedger,
    task: EvoTask,
    sample_id: str,
    batch_id: str,
    source: str,
    menu_id: str,
    value_snapshot_id: str,
    statistics_context: str,
    value_function: Callable[[tuple[float, ...], str], float],
    seed: int,
    value_batch_function: (
        Callable[[Sequence[tuple[float, ...]], str], Sequence[float]] | None
    ) = None,
    value_fitted: bool = False,
    controller_deadline_ms: int = 120_000,
    forced_first_action: GraphAction | None = None,
    pair_id: str | None = None,
    candidate_id: str | None = None,
) -> EvoTrajectory:
    reference = source != "current"
    behavior_id = policy.reference_id if reference else policy.actor_id
    records: list[DecisionRecord] = []
    previous_value: float | None = None
    if type(value_fitted) is not bool:
        raise TypeError("value_fitted must be a boolean")
    batched: dict[tuple[tuple[float, ...], ...], tuple[float, ...]] = {}

    def diagnose(state: Any, features: tuple[float, ...], current_value: float) -> str:
        def values(rows: Sequence[tuple[float, ...]]) -> Sequence[float]:
            key = tuple(tuple(row) for row in rows)
            if key not in batched:
                batched.clear()
                batched[key] = tuple(
                    float(item)
                    for item in (
                        value_batch_function(rows, task.family)
                        if value_batch_function is not None
                        else [value_function(tuple(row), task.family) for row in rows]
                    )
                )
            return batched[key]

        return diagnosis_text(
            features,
            state,
            decision=len(records),
            value=current_value,
            start_value=records[0].value_estimate if records else None,
            previous_value=previous_value,
            last_action=json.loads(records[-1].action_json)["kind"] if records else None,
            fitted=value_fitted,
            values=values,
        )

    while not runtime.stopped:
        actions = runtime.legal_actions()
        if not actions:
            raise RuntimeError("no legal graph continuation; episode has no fabricated reward")
        initial_menu = policy.menu(tuple(action.to_value() for action in actions))
        forced = forced_first_action if not records else None
        features = runtime.features()
        value = float(value_function(features, task.family))

        def render(
            current_actions: tuple[GraphAction, ...],
            current_value: float,
            old_value: float | None,
            current_features: tuple[float, ...],
            finalizing: bool = False,
        ) -> str:
            state = runtime.public_state()
            if finalizing:
                state["controller_budget_finalization"] = True
            text = str(
                canonical_json(
                    {
                        "task_family": task.family,
                        "execution": controller_projection(state),
                        "reference_value": current_value,
                        "reference_value_change": (
                            0.0 if old_value is None else current_value - old_value
                        ),
                        "legal_actions": [action.to_value() for action in current_actions],
                    }
                )
            )
            return text + diagnose(state, current_features, current_value)

        preview = policy.encode_prompt(render(actions, value, previous_value, features))
        reservation_id = f"{sample_id}:controller:{len(records)}"
        charged = forced is None and len(actions) > 1
        budget_finalization = False
        if charged:
            maximum = BudgetVector(
                input_tokens=min(
                    policy.context_window - policy.max_action_tokens, len(preview) + 256
                ),
                output_tokens=max(map(len, initial_menu.paths)),
                model_calls=1,
                agent_turns=1,
                wall_time_milliseconds=controller_deadline_ms,
            )
            if not can_reserve(ledger, maximum):
                charged = False
                budget_finalization = True
                actions = (runtime.finalization_action(),)
            else:
                ledger.reserve(
                    BudgetReservation(
                        reservation_id, ledger.run_id, ledger.attempt_id, sample_id, maximum
                    )
                )
                actions = runtime.legal_actions()
                if not actions:
                    ledger.settle(BudgetSettlement(reservation_id, BudgetVector()))
                    charged = False
                    budget_finalization = True
                    actions = (runtime.finalization_action(),)
        state = runtime.public_state()
        if budget_finalization:
            state["controller_budget_finalization"] = True
        state_json = canonical_json(state)
        state_id = stable_hash(state)
        features = runtime.features()
        value = float(value_function(features, task.family))
        menu = policy.menu(tuple(action.to_value() for action in actions))
        prompt_ids = policy.encode_prompt(
            render(actions, value, previous_value, features, budget_finalization)
        )
        if charged and len(prompt_ids) > maximum.input_tokens:
            raise RuntimeError("rendered controller input exceeds reserved envelope")
        started = time.monotonic()
        model_called = False
        if forced is not None:
            action_json = canonical_json(forced.to_value())
            if action_json not in menu.actions:
                raise ValueError("paired initial intervention is not a legal first action")
            tokens = menu.paths[menu.actions.index(action_json)]
        elif len(menu.paths) == 1:
            tokens = menu.paths[0]
        else:
            tokens = policy.sample(prompt_ids, menu, reference=reference, seed=seed + len(records))
            model_called = True
        elapsed_ms = max(0, int((time.monotonic() - started) * 1000))
        if tokens not in menu.paths:
            raise ValueError("policy returned an action outside the exact legal grammar")
        if charged:
            actual = (
                BudgetVector(
                    input_tokens=len(prompt_ids),
                    output_tokens=len(tokens),
                    model_calls=1,
                    agent_turns=1,
                    wall_time_milliseconds=min(elapsed_ms, controller_deadline_ms),
                )
                if model_called
                else BudgetVector()
            )
            ledger.settle(BudgetSettlement(reservation_id, actual))
        action_json = menu.actions[menu.paths.index(tokens)]
        action = GraphAction.from_value(json.loads(action_json))
        records.append(
            DecisionRecord(
                state_id=state_id,
                state_json=state_json,
                action_json=action_json,
                prompt_ids=prompt_ids,
                action_token_ids=tokens,
                legal_token_paths=menu.paths,
                features=features,
                reference_encoding=(),
                value_estimate=value,
                forced=forced is not None,
            )
        )
        await runtime.apply(action)
        previous_value = value
    ledger.assert_fully_settled()
    if runtime.reward is None or runtime.output is None:
        raise RuntimeError("STOP did not produce a terminal result")
    return EvoTrajectory(
        sample_id=sample_id,
        batch_id=batch_id,
        task=task,
        source=source,
        behavior_policy_id=behavior_id,
        reference_id=policy.reference_id,
        executor_id=str(runtime.public_state()["executor_identity"]),
        menu_id=menu_id,
        value_snapshot_id=value_snapshot_id,
        statistics_context=statistics_context,
        decisions=tuple(records),
        terminal_state_json=canonical_json(runtime.public_state()),
        reward=runtime.reward,
        output=runtime.output,
        usage_json=canonical_json(ledger.settled.to_value()),
        pair_id=pair_id,
        candidate_id=candidate_id,
    )
