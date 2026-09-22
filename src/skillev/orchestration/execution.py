"""Online team execution with explicit graph effects and exact shared accounting.

Only committed, healthy boundaries can be restored. A failed external call may
have changed an interactive world: the runtime is poisoned, not rolled back or
retried. STOP remains legal when node execution budget is exhausted; there is
no invented answer/reward if the budget cannot produce even one output.

Communication reaches a node only when it executes (paper Fig. 1/2: Solve ->
Check -> feedback -> rerun). Every execution receives the current outputs of
all nodes with an edge into it, so a feedback edge is delivered on the
target's next RERUN_AGENT/BIND_SKILL, while a revise edge executes the target
at once. A node of any role other than the first declared (solving) role
checks work: each of its executions also receives the available draft, i.e.
the selected output node's current text, else the most recently executed other
live node's text, as an explicit ``draft`` message naming its source node.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from skillev.contracts.canonical import JsonValue, normalize_json, stable_hash
from skillev.runtime.budget_ledger import BudgetExceededError, BudgetLedger
from skillev.runtime.contracts import BudgetReservation, BudgetSettlement, BudgetVector

from .actions import EDGE_PROTOCOLS, GraphAction, GraphActionKind
from .budget import can_reserve, total_token_budget
from .evosteer_features import FEATURE_NAMES, FEATURE_VERSION, extract_features
from .graph import (
    GraphState,
    NodeExecutionRequest,
    NodeExecutionResult,
    NodeExecutor,
    NodeState,
    ProtocolEdge,
    RoleSpec,
)

# @3: checking roles receive a draft message; config roles keep declared order.
SNAPSHOT_VERSION = "evosteer-graph-runtime@3"
# A message protocol, not an edge protocol: it never enters the action surface.
DRAFT_MESSAGE_PROTOCOL = "draft"
TerminalEvaluator = Callable[[str], float | Awaitable[float]]


def _skill_label(body: str) -> dict[str, JsonValue]:
    """The skill's own name and trigger line, for the menu; never the procedure.

    An authored skill is a JSON object with "name" and "trigger" fields that say
    what it is for; those two fields label the slot. Any other body keeps Remark
    C.3 exactly: identifier and hash only, so no procedure text can leak.
    """
    try:
        value = json.loads(body)
    except ValueError:
        return {}
    if not isinstance(value, dict):
        return {}
    label: dict[str, JsonValue] = {}
    name, trigger = value.get("name"), value.get("trigger")
    if isinstance(name, str) and name.strip():
        label["name"] = name.strip()[:120]
    if isinstance(trigger, str) and trigger.strip():
        label["trigger"] = trigger.strip()[:240]
    return label


def _draft_source(
    *,
    node_id: str,
    role_id: str,
    primary_role_id: str,
    nodes: Mapping[str, NodeState],
    output_id: str | None,
    history: Sequence[Mapping[str, Any]],
    inbound: Collection[str],
) -> str | None:
    """Pick the draft a checking node executes against; None for the solving role.

    The selected output is the team's current answer, so it is what a check is
    about. Before any selection, the most recently executed live node of the
    solving role (by the public history) is the freshest draft, so a checker
    checks a solution rather than another checker's verdict; only when no such
    node exists does the most recently executed live node of any role serve. A
    node never receives its own text (that is ``previous_output``), and a
    source already delivered through an edge into the node is not repeated.
    """
    if role_id == primary_role_id:
        return None
    source = output_id if output_id is not None and output_id != node_id else None
    if source is None:
        executed = []
        for event in reversed(history):
            observation = event.get("observation")
            if not isinstance(observation, Mapping) or "node_result" not in observation:
                continue
            candidate = observation.get("node_id")
            if isinstance(candidate, str) and candidate != node_id and candidate in nodes:
                executed.append(candidate)
        solving = [c for c in executed if nodes[c].role_id == primary_role_id]
        source = (solving or executed or [None])[0]
    if source is None or source not in nodes or source in inbound:
        return None
    return source


def _draft_message(source: NodeState, target_id: str) -> dict[str, JsonValue]:
    # Same shape as an edge message so executors render it uniformly, and the
    # recorded node request names the exact source node and output version.
    return {
        "source_id": source.node_id,
        "target_id": target_id,
        "protocol": DRAFT_MESSAGE_PROTOCOL,
        "source_output_version": source.output_version,
        "body": source.output,
    }


class IllegalGraphActionError(ValueError):
    """Rejected before execution: no graph, world or budget mutation occurred."""


class RuntimePoisonedError(RuntimeError):
    """An external effect failed; continuing could hide an unknown world change."""


class _NodeCallNotAdmittedError(IllegalGraphActionError):
    """Concurrent resource use consumed capacity before any node call began."""


@dataclass(frozen=True, slots=True)
class ExecutionReceipt:
    sequence: int
    action: GraphAction
    before_state_id: str
    after_state_id: str
    observation: dict[str, JsonValue]
    features: tuple[float, ...]
    stopped: bool
    reward: float | None

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "sequence": self.sequence,
            "action": self.action.to_value(),
            "before_state_id": self.before_state_id,
            "after_state_id": self.after_state_id,
            "observation": self.observation,
            "features": list(self.features),
            "feature_version": FEATURE_VERSION,
            "stopped": self.stopped,
            "reward": self.reward,
        }


class GraphRuntime:
    """One team, one serialized world, immutable role/skill/executor snapshots.

    Team size and episode length have no method-level ceiling. Optional numeric
    limits are debug controls; with a debug horizon the final slots are reserved
    for SET_OUTPUT/STOP. IDs increase monotonically after drops. Node calls and
    controller decisions use the same finite ledger; the collector must charge
    every nondeterministic controller choice and finalize when it cannot reserve.
    The first declared role is the solving role; every other role receives the
    available draft on each of its executions (see the module docstring).
    """

    def __init__(
        self,
        *,
        task_prompt: str,
        roles: Sequence[RoleSpec] | Mapping[str, RoleSpec],
        skills: Mapping[str, str],
        executor: NodeExecutor,
        ledger: BudgetLedger,
        evaluator: TerminalEvaluator,
        max_nodes: int | None = None,
        max_actions: int | None = None,
        seed: int = 0,
        runtime_id: str = "evosteer",
    ) -> None:
        if type(task_prompt) is not str or not task_prompt.strip():
            raise ValueError("task_prompt must be nonempty public text")
        if type(runtime_id) is not str or not runtime_id.strip():
            raise ValueError("runtime_id must be nonempty text")
        if max_nodes is not None and (type(max_nodes) is not int or max_nodes < 1):
            raise ValueError("debug max_nodes must be positive or None")
        if max_actions is not None and (type(max_actions) is not int or max_actions < 3):
            raise ValueError("debug max_actions must allow ADD_AGENT, SET_OUTPUT, STOP or be None")
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        role_values = tuple(roles.values()) if isinstance(roles, Mapping) else tuple(roles)
        if not role_values or any(not isinstance(role, RoleSpec) for role in role_values):
            raise ValueError("roles must contain RoleSpec objects")
        if len({role.role_id for role in role_values}) != len(role_values):
            raise ValueError("role IDs must be unique")
        if isinstance(roles, Mapping) and set(roles) != {role.role_id for role in role_values}:
            raise ValueError("role mapping keys must equal role IDs")
        if any(
            type(key) is not str or not key.strip() or type(body) is not str or not body.strip()
            for key, body in skills.items()
        ):
            raise ValueError("skills must map nonempty IDs to nonempty public bodies")
        identity = executor.frozen_identity
        if type(identity) is not str or not identity.strip():
            raise ValueError("executor must expose its pinned frozen identity")
        if not isinstance(ledger, BudgetLedger) or not callable(evaluator):
            raise TypeError("ledger and evaluator are required")
        self.task_prompt = task_prompt
        self.runtime_id = runtime_id
        self.max_nodes = max_nodes
        self.max_actions = max_actions
        self.seed = seed
        # Declared order is configuration (roles[0] solves; paired trials force
        # it first), so snapshots keep it; the public role list stays sorted.
        self._declared_roles = role_values
        self._primary_role_id = role_values[0].role_id
        self._roles = {role.role_id: role for role in sorted(role_values, key=lambda r: r.role_id)}
        self._skills = dict(sorted(skills.items()))
        self._executor = executor
        self._executor_identity = identity
        self._ledger = ledger
        self._evaluator = evaluator
        self._nodes: dict[str, NodeState] = {}
        self._edges: list[ProtocolEdge] = []
        self._output_node_id: str | None = None
        self._next_node_index = 0
        self._executions = 0
        self._history: list[dict[str, JsonValue]] = []
        self._node_usage = BudgetVector()
        self._stopped = False
        self._reward: float | None = None
        self._poisoned = False
        self._busy = False
        self._lock = asyncio.Lock()

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def reward(self) -> float | None:
        return self._reward

    @property
    def output(self) -> str | None:
        node = self._nodes.get(self._output_node_id or "")
        return None if node is None else node.output

    @property
    def graph(self) -> GraphState:
        return GraphState(
            tuple(self._nodes[key] for key in sorted(self._nodes)),
            tuple(sorted(self._edges, key=lambda edge: (edge.source_id, edge.target_id))),
            self._output_node_id,
        )

    @property
    def history(self) -> tuple[dict[str, JsonValue], ...]:
        # All public projections are owned copies; caller mutation cannot alter state.
        return tuple(cast(dict[str, JsonValue], normalize_json(event)) for event in self._history)

    @property
    def node_usage(self) -> BudgetVector:
        return self._node_usage

    @property
    def state_id(self) -> str:
        return stable_hash(self.public_state())

    def _environment_state(self) -> dict[str, JsonValue] | None:
        public_state = getattr(self._executor, "public_environment_state", None)
        if not callable(public_state):
            return None
        result = normalize_json(public_state())
        if not isinstance(result, dict):
            raise TypeError("executor public_environment_state must return a JSON object")
        return cast(dict[str, JsonValue], result)

    def public_state(self) -> dict[str, JsonValue]:
        state = normalize_json(
            {
                "format": SNAPSHOT_VERSION,
                "runtime_id": self.runtime_id,
                "task_prompt": self.task_prompt,
                "graph": self.graph.to_value(),
                # Declared order, with the solving role named: every other role
                # also receives the current draft when it executes.
                "roles": [role.to_value() for role in self._declared_roles],
                "primary_role_id": self._primary_role_id,
                # Paper Remark C.3 exposes slot identifiers, not inline skill text.
                # Bodies are supplied only when a bound node actually executes.
                # Disclosed deviation: each slot also carries the skill's own name
                # and trigger line, so the orchestrator can tell one slot from
                # another; with identifiers alone a binding is a blind guess.
                "skill_menu": [
                    {"skill_id": key, "content_hash": stable_hash(body), **_skill_label(body)}
                    for key, body in self._skills.items()
                ],
                "history": self._history,
                "executor_identity": self._executor_identity,
                "actions_remaining": (
                    None if self.max_actions is None else self.max_actions - len(self._history)
                ),
                "debug_limits": {"max_nodes": self.max_nodes, "max_actions": self.max_actions},
                "environment": self._environment_state(),
                "feature_version": FEATURE_VERSION,
                "feature_names": list(FEATURE_NAMES),
                "execution_features": list(self.features()),
                "budget_available": self._ledger.available.to_value(),
                "total_token_budget": total_token_budget(self._ledger),
                "node_usage": self._node_usage.to_value(),
                "stopped": self._stopped,
                "poisoned": self._poisoned,
            }
        )
        return cast(dict[str, JsonValue], state)

    def features(self) -> tuple[float, ...]:
        return extract_features(
            graph=self.graph.to_value(),
            history=self._history,
            max_nodes=self.max_nodes,
            max_actions=self.max_actions,
            role_count=len(self._roles),
            skill_count=len(self._skills),
            used=self._ledger.settled,
            cap=self._ledger.cap,
            total_token_cap=getattr(self._ledger, "total_token_cap", None),
            environment=self._environment_state(),
        )

    def _can_execute(self, role_id: str) -> bool:
        return can_reserve(self._ledger, self._roles[role_id].model_maximum)

    def legal_actions(self) -> tuple[GraphAction, ...]:
        """Finite deterministic surface; suitable for an exact token prefix trie."""
        if self._stopped or self._poisoned or self._busy:
            return ()
        remaining = None if self.max_actions is None else self.max_actions - len(self._history)
        if remaining is not None and remaining <= 0:
            return ()
        stop = [GraphAction(GraphActionKind.STOP)] if self.output is not None else []
        if remaining == 1:
            return tuple(stop)
        select = [
            GraphAction(GraphActionKind.SET_OUTPUT, node_id=node_id)
            for node_id in sorted(self._nodes)
            if node_id != self._output_node_id
        ]
        if remaining == 2 and self.output is None:
            return tuple(select)
        choices = [*stop, *select]
        if self.max_nodes is None or len(self._nodes) < self.max_nodes:
            for role_id in self._roles:
                if self._can_execute(role_id):
                    for skill_id in (None, *self._skills):
                        choices.append(
                            GraphAction(
                                GraphActionKind.ADD_AGENT,
                                node_id=f"n{self._next_node_index}",
                                role_id=role_id,
                                skill_id=skill_id,
                            )
                        )
        occupied = {(edge.source_id, edge.target_id) for edge in self._edges}
        for node_id, node in sorted(self._nodes.items()):
            # Do not allow deletion to strand a team that cannot create/select/stop.
            can_replace_last = (remaining is None or remaining >= 4) and any(
                self._can_execute(role_id) for role_id in self._roles
            )
            if node_id != self._output_node_id and (len(self._nodes) > 1 or can_replace_last):
                choices.append(GraphAction(GraphActionKind.DROP_AGENT, node_id=node_id))
            if self._can_execute(node.role_id):
                choices.append(GraphAction(GraphActionKind.RERUN_AGENT, node_id=node_id))
                choices.extend(
                    GraphAction(GraphActionKind.BIND_SKILL, node_id=node_id, skill_id=skill)
                    for skill in self._skills
                    if skill not in node.skill_ids
                )
            for target_id, target in sorted(self._nodes.items()):
                if node_id == target_id or (node_id, target_id) in occupied:
                    continue
                for protocol in EDGE_PROTOCOLS:
                    if protocol == "feedback" or self._can_execute(target.role_id):
                        choices.append(
                            GraphAction(
                                GraphActionKind.ADD_EDGE,
                                source_id=node_id,
                                target_id=target_id,
                                protocol=protocol,
                            )
                        )
        return tuple(sorted(choices, key=lambda action: action.text))

    def _messages(
        self,
        target_id: str,
        role_id: str,
        extra: ProtocolEdge | None = None,
    ) -> tuple[dict[str, JsonValue], ...]:
        # Every execution (ADD, RERUN, BIND, revise) reads each inbound edge's
        # source at its current output version: queued feedback is delivered here.
        edges = [edge for edge in self._edges if edge.target_id == target_id]
        if extra is not None:
            edges.append(extra)
        messages = [self._message(edge) for edge in sorted(edges, key=lambda e: e.source_id)]
        draft = _draft_source(
            node_id=target_id,
            role_id=role_id,
            primary_role_id=self._primary_role_id,
            nodes=self._nodes,
            output_id=self._output_node_id,
            history=self._history,
            inbound={edge.source_id for edge in edges},
        )
        if draft is not None:
            messages.insert(0, _draft_message(self._nodes[draft], target_id))
        return tuple(messages)

    def _message(self, edge: ProtocolEdge) -> dict[str, JsonValue]:
        source = self._nodes[edge.source_id]
        return {
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "protocol": edge.protocol,
            "source_output_version": source.output_version,
            "body": source.output,
        }

    async def _execute_node(
        self,
        *,
        node_id: str,
        role_id: str,
        skill_ids: tuple[str, ...],
        previous: NodeState | None,
        extra_edge: ProtocolEdge | None = None,
    ) -> tuple[NodeState, dict[str, JsonValue]]:
        if self._executor.frozen_identity != self._executor_identity:
            raise RuntimeError("frozen executor identity changed")
        role = self._roles[role_id]
        sequence = len(self._history) + 1
        execution_index = self._executions + 1
        request = NodeExecutionRequest(
            runtime_id=self.runtime_id,
            node_id=node_id,
            role=role,
            task_prompt=self.task_prompt,
            skills=tuple((skill_id, self._skills[skill_id]) for skill_id in skill_ids),
            messages=self._messages(node_id, role_id, extra_edge),
            previous_output=None if previous is None else previous.output,
            execution_index=execution_index,
            seed=int(
                stable_hash(
                    {"seed": self.seed, "execution_index": execution_index, "node_id": node_id}
                ).split(":", 1)[-1][:16],
                16,
            ),
        )
        reservation = BudgetReservation(
            reservation_id=f"{self.runtime_id}:graph:{sequence}:node:{execution_index}",
            run_id=self._ledger.run_id,
            attempt_id=self._ledger.attempt_id,
            invocation_id=self.runtime_id,
            maximum=role.model_maximum,
        )
        try:
            self._ledger.reserve(reservation)
        except BudgetExceededError as error:
            raise _NodeCallNotAdmittedError(
                "shared node budget was consumed before dispatch"
            ) from error
        # Record the exact input before handing owned mutable projections to external code.
        request_value = normalize_json(request.to_value())
        result = await self._executor.execute(request)
        if not isinstance(result, NodeExecutionResult):
            raise TypeError("executor returned an incompatible result")
        self._ledger.settle(BudgetSettlement(reservation.reservation_id, result.usage))
        self._node_usage = self._node_usage.add(result.usage)
        if self._executor.frozen_identity != self._executor_identity:
            raise RuntimeError("frozen executor identity changed during execution")
        self._executions = execution_index
        node = NodeState(
            node_id,
            role_id,
            skill_ids,
            result.output,
            1 if previous is None else previous.output_version + 1,
        )
        observation: dict[str, JsonValue] = {
            "node_request": request_value,
            "node_result": result.to_value(),
            "node_id": node_id,
            "output_version": node.output_version,
            "output_changed": previous is not None and previous.output != result.output,
            "reservation_id": reservation.reservation_id,
        }
        return node, observation

    async def apply(self, action: GraphAction) -> ExecutionReceipt:
        """Apply one admitted action, with at most one non-recursive node execution."""
        async with self._lock:
            if self._poisoned:
                raise RuntimePoisonedError("runtime has an uncertain external effect")
            if not isinstance(action, GraphAction) or action not in self.legal_actions():
                raise IllegalGraphActionError(
                    "action is not in the current finite legal action set"
                )
            before = self.state_id
            self._busy = True
            observation: dict[str, JsonValue] = {}
            try:
                if action.kind is GraphActionKind.ADD_AGENT:
                    node_id, role_id = str(action.node_id), str(action.role_id)
                    node, observation = await self._execute_node(
                        node_id=node_id,
                        role_id=role_id,
                        skill_ids=() if action.skill_id is None else (action.skill_id,),
                        previous=None,
                    )
                    self._nodes[node_id] = node
                    self._next_node_index += 1
                elif action.kind is GraphActionKind.ADD_EDGE:
                    edge = ProtocolEdge(
                        str(action.source_id), str(action.target_id), str(action.protocol)
                    )
                    observation["delivered_message"] = self._message(edge)
                    if edge.protocol == "revise":
                        previous = self._nodes[edge.target_id]
                        node, execution = await self._execute_node(
                            node_id=previous.node_id,
                            role_id=previous.role_id,
                            skill_ids=previous.skill_ids,
                            previous=previous,
                            extra_edge=edge,
                        )
                        self._nodes[node.node_id] = node
                        observation.update(execution)
                    self._edges.append(edge)
                elif action.kind in {GraphActionKind.BIND_SKILL, GraphActionKind.RERUN_AGENT}:
                    previous = self._nodes[str(action.node_id)]
                    skill_ids = previous.skill_ids
                    if action.kind is GraphActionKind.BIND_SKILL:
                        skill_ids = tuple(sorted((*skill_ids, str(action.skill_id))))
                    node, observation = await self._execute_node(
                        node_id=previous.node_id,
                        role_id=previous.role_id,
                        skill_ids=skill_ids,
                        previous=previous,
                    )
                    self._nodes[node.node_id] = node
                elif action.kind is GraphActionKind.SET_OUTPUT:
                    self._output_node_id = action.node_id
                    observation["selected_output_node"] = action.node_id
                elif action.kind is GraphActionKind.DROP_AGENT:
                    node_id = str(action.node_id)
                    del self._nodes[node_id]
                    self._edges = [
                        e for e in self._edges if node_id not in (e.source_id, e.target_id)
                    ]
                    observation["dropped_node"] = node_id
                elif action.kind is GraphActionKind.STOP:
                    assert self.output is not None
                    evaluated = self._evaluator(self.output)
                    reward = await evaluated if inspect.isawaitable(evaluated) else evaluated
                    if isinstance(reward, bool) or not isinstance(reward, float | int):
                        raise TypeError("terminal evaluator must return a number")
                    if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
                        raise ValueError("terminal reward must lie in [0,1]")
                    self._reward = float(reward)
                    self._stopped = True
                    observation.update(
                        {
                            "final_output": self.output,
                            "output_node_id": self._output_node_id,
                        }
                    )
                event = normalize_json(
                    {
                        "sequence": len(self._history) + 1,
                        "action": action.to_value(),
                        "observation": observation,
                    }
                )
                self._history.append(cast(dict[str, JsonValue], event))
            except _NodeCallNotAdmittedError:
                # reserve() rejected before dispatch and did not debit the ledger.
                raise
            except BaseException as error:
                # Cancellation can also occur after the world has changed.
                self._poisoned = True
                self._history.append(
                    {
                        "sequence": len(self._history) + 1,
                        "action": action.to_value(),
                        "observation": {
                            "status": "infrastructure-failure",
                            "error_type": type(error).__name__,
                        },
                    }
                )
                raise
            finally:
                self._busy = False
            return ExecutionReceipt(
                len(self._history),
                action,
                before,
                self.state_id,
                cast(dict[str, JsonValue], normalize_json(observation)),
                self.features(),
                self._stopped,
                self._reward,
            )

    def snapshot(self) -> dict[str, JsonValue]:
        """Capture a committed boundary; restore requires the same settled ledger."""
        if self._busy or self._poisoned:
            raise RuntimePoisonedError("only healthy committed runtime states can be checkpointed")
        self._ledger.assert_fully_settled()
        value: dict[str, object] = {
            "format": SNAPSHOT_VERSION,
            "config": {
                "task_prompt": self.task_prompt,
                "runtime_id": self.runtime_id,
                "max_nodes": self.max_nodes,
                "max_actions": self.max_actions,
                "seed": self.seed,
                "roles": [role.to_value() for role in self._declared_roles],
                "skills": self._skills,
                "executor_identity": self._executor_identity,
            },
            "graph": self.graph.to_value(),
            "history": self._history,
            "next_node_index": self._next_node_index,
            "executions": self._executions,
            "node_usage": self._node_usage.to_value(),
            "stopped": self._stopped,
            "reward": self._reward,
            "environment": self._environment_state(),
            "ledger": {
                "run_id": self._ledger.run_id,
                "attempt_id": self._ledger.attempt_id,
                "cap": self._ledger.cap.to_value(),
                "settled": self._ledger.settled.to_value(),
                "total_token_budget": total_token_budget(self._ledger),
                "reservation_ids": [
                    entry.reservation.reservation_id for entry in self._ledger.entries
                ],
            },
        }
        result = normalize_json({**value, "snapshot_hash": stable_hash(value)})
        return cast(dict[str, JsonValue], result)

    @classmethod
    def from_snapshot(
        cls,
        value: object,
        *,
        executor: NodeExecutor,
        ledger: BudgetLedger,
        evaluator: TerminalEvaluator,
    ) -> GraphRuntime:
        """Restore without calls, charges or evaluator invocation; never infer missing state."""
        if not isinstance(value, dict) or set(value) != {
            "format",
            "config",
            "graph",
            "history",
            "next_node_index",
            "executions",
            "node_usage",
            "stopped",
            "reward",
            "environment",
            "ledger",
            "snapshot_hash",
        }:
            raise ValueError("invalid graph runtime snapshot fields")
        data: Any = normalize_json(value)
        if data["format"] != SNAPSHOT_VERSION or data["snapshot_hash"] != stable_hash(
            {key: item for key, item in data.items() if key != "snapshot_hash"}
        ):
            raise ValueError("graph runtime snapshot identity mismatch")
        ledger.assert_fully_settled()
        expected_ledger = {
            "run_id": ledger.run_id,
            "attempt_id": ledger.attempt_id,
            "cap": ledger.cap.to_value(),
            "settled": ledger.settled.to_value(),
            "total_token_budget": total_token_budget(ledger),
            "reservation_ids": [entry.reservation.reservation_id for entry in ledger.entries],
        }
        if data["ledger"] != expected_ledger:
            raise ValueError("restore requires the exact saved ledger including controller charges")
        config = dict(data["config"])
        identity = config.pop("executor_identity")
        if identity != executor.frozen_identity:
            raise ValueError("snapshot executor identity does not match")
        config["roles"] = tuple(RoleSpec.from_value(role) for role in config["roles"])
        runtime = cls(**config, executor=executor, ledger=ledger, evaluator=evaluator)
        if data["environment"] != runtime._environment_state():
            raise ValueError("restore requires the same observed external environment state")
        history = data["history"]
        if not isinstance(history, list) or (
            runtime.max_actions is not None and len(history) > runtime.max_actions
        ):
            raise ValueError("snapshot history exceeds action horizon")
        for index, event in enumerate(history, 1):
            if not isinstance(event, dict) or set(event) != {"sequence", "action", "observation"}:
                raise ValueError("invalid history event")
            if event["sequence"] != index or not isinstance(event["observation"], dict):
                raise ValueError("history sequence is not contiguous")
            GraphAction.from_value(event["action"])
            if event["observation"].get("status") == "infrastructure-failure":
                raise ValueError("cannot restore an uncertain external effect")
        graph = data["graph"]
        if not isinstance(graph, dict) or set(graph) != {"nodes", "edges", "output_node_id"}:
            raise ValueError("invalid saved graph")
        nodes: dict[str, NodeState] = {}
        for raw in graph["nodes"]:
            if not isinstance(raw, dict) or set(raw) != {
                "node_id",
                "role_id",
                "skill_ids",
                "output",
                "output_version",
            }:
                raise ValueError("invalid saved node")
            node = NodeState(**{**raw, "skill_ids": tuple(raw["skill_ids"])})
            if node.node_id in nodes or node.role_id not in runtime._roles:
                raise ValueError("saved node identity is inconsistent")
            if (
                type(node.output) is not str
                or type(node.output_version) is not int
                or node.output_version < 1
            ):
                raise ValueError("saved node requires a versioned output")
            if (
                tuple(sorted(set(node.skill_ids))) != node.skill_ids
                or not set(node.skill_ids) <= runtime._skills.keys()
            ):
                raise ValueError("saved node skill identity is inconsistent")
            nodes[node.node_id] = node
        edges: list[ProtocolEdge] = []
        endpoints: set[tuple[str, str]] = set()
        for raw in graph["edges"]:
            if not isinstance(raw, dict) or set(raw) != {"source_id", "target_id", "protocol"}:
                raise ValueError("invalid saved edge")
            edge = ProtocolEdge(**raw)
            if (
                edge.source_id not in nodes
                or edge.target_id not in nodes
                or edge.source_id == edge.target_id
                or edge.protocol not in EDGE_PROTOCOLS
            ):
                raise ValueError("saved edge references an invalid endpoint/protocol")
            if (edge.source_id, edge.target_id) in endpoints:
                raise ValueError("duplicate saved edge")
            endpoints.add((edge.source_id, edge.target_id))
            edges.append(edge)
        output_id = graph["output_node_id"]
        if output_id is not None and output_id not in nodes:
            raise ValueError("saved output node is missing")
        if runtime.max_nodes is not None and len(nodes) > runtime.max_nodes:
            raise ValueError("saved graph exceeds live node cap")
        for name in ("next_node_index", "executions"):
            if type(data[name]) is not int or not 0 <= data[name] <= len(history):
                raise ValueError("invalid saved execution counters")
        valid_ids = {f"n{i}" for i in range(data["next_node_index"])}
        if any(node_id not in valid_ids for node_id in nodes):
            raise ValueError("saved node IDs exceed the allocation cursor")
        stopped, reward = data["stopped"], data["reward"]
        if type(stopped) is not bool or (
            stopped
            and (not history or history[-1]["action"]["kind"] != "STOP" or output_id is None)
        ):
            raise ValueError("invalid saved terminal state")
        if stopped:
            if (
                isinstance(reward, bool)
                or not isinstance(reward, float | int)
                or not math.isfinite(reward)
                or not 0 <= reward <= 1
            ):
                raise ValueError("invalid saved terminal reward")
        elif reward is not None or any(event["action"]["kind"] == "STOP" for event in history):
            raise ValueError("unfinished snapshot contains a terminal result")
        runtime._nodes, runtime._edges = nodes, edges
        runtime._output_node_id, runtime._history = output_id, history
        runtime._next_node_index, runtime._executions = data["next_node_index"], data["executions"]
        runtime._node_usage = BudgetVector.from_value(data["node_usage"])
        if not runtime._node_usage.fits_within(ledger.settled):
            raise ValueError("node usage exceeds saved total accounting")
        runtime._stopped, runtime._reward = stopped, reward
        runtime._validate_saved_history()
        return runtime

    def _validate_saved_history(self) -> None:
        """Recompute the public graph from receipts without redoing external effects."""
        nodes: dict[str, NodeState] = {}
        edges: list[ProtocolEdge] = []
        output_id: str | None = None
        next_id = executions = 0
        usage = BudgetVector()
        ledger_entries = {entry.reservation.reservation_id: entry for entry in self._ledger.entries}
        for index, event in enumerate(self._history, 1):
            action = GraphAction.from_value(event["action"])
            observation: Any = event["observation"]
            extra_edge: ProtocolEdge | None = None
            execute_id: str | None = None
            role_id: str | None = None
            skill_ids: tuple[str, ...] = ()
            expected_observation: dict[str, JsonValue] = {}
            if action.kind is GraphActionKind.ADD_AGENT:
                if action.node_id != f"n{next_id}" or (
                    self.max_nodes is not None and len(nodes) >= self.max_nodes
                ):
                    raise ValueError("history has an invalid node allocation")
                execute_id, role_id = action.node_id, action.role_id
                if role_id not in self._roles:
                    raise ValueError("history has an unknown role")
                skill_ids = () if action.skill_id is None else (action.skill_id,)
                next_id += 1
            elif action.kind is GraphActionKind.ADD_EDGE:
                if action.source_id not in nodes or action.target_id not in nodes:
                    raise ValueError("history edge has a missing endpoint")
                if any(
                    (edge.source_id, edge.target_id) == (action.source_id, action.target_id)
                    for edge in edges
                ):
                    raise ValueError("history contains a duplicate edge")
                extra_edge = ProtocolEdge(
                    str(action.source_id), str(action.target_id), str(action.protocol)
                )
                source = nodes[str(action.source_id)]
                expected_observation["delivered_message"] = {
                    **extra_edge.to_value(),
                    "body": source.output,
                    "source_output_version": source.output_version,
                }
                if action.protocol == "revise":
                    execute_id = action.target_id
                    role_id = nodes[str(execute_id)].role_id
                    skill_ids = nodes[str(execute_id)].skill_ids
            elif action.kind in {GraphActionKind.RERUN_AGENT, GraphActionKind.BIND_SKILL}:
                if action.node_id not in nodes:
                    raise ValueError("history executes a missing node")
                previous = nodes[str(action.node_id)]
                execute_id, role_id, skill_ids = (
                    previous.node_id,
                    previous.role_id,
                    previous.skill_ids,
                )
                if action.kind is GraphActionKind.BIND_SKILL:
                    if action.skill_id in skill_ids:
                        raise ValueError("history repeats an existing skill binding")
                    skill_ids = tuple(sorted((*skill_ids, str(action.skill_id))))
            elif action.kind is GraphActionKind.SET_OUTPUT:
                if action.node_id not in nodes or action.node_id == output_id:
                    raise ValueError("history selects an invalid output")
                output_id = action.node_id
                expected_observation["selected_output_node"] = output_id
            elif action.kind is GraphActionKind.DROP_AGENT:
                if action.node_id not in nodes or action.node_id == output_id:
                    raise ValueError("history drops an invalid node")
                del nodes[str(action.node_id)]
                edges = [e for e in edges if action.node_id not in (e.source_id, e.target_id)]
                expected_observation["dropped_node"] = action.node_id
            elif action.kind is GraphActionKind.STOP:
                if output_id not in nodes or index != len(self._history) or not self.stopped:
                    raise ValueError("history stops without a valid terminal output")
                expected_observation = {
                    "final_output": nodes[str(output_id)].output,
                    "output_node_id": output_id,
                }

            if execute_id is not None:
                if not set(skill_ids) <= self._skills.keys():
                    raise ValueError("history binds an unavailable skill")
                prev_executed_node = nodes.get(execute_id)
                incoming = [e for e in edges if e.target_id == execute_id]
                if extra_edge is not None:
                    incoming.append(extra_edge)
                messages_list: list[dict[str, JsonValue]] = [
                    {
                        **edge.to_value(),
                        "body": nodes[edge.source_id].output,
                        "source_output_version": nodes[edge.source_id].output_version,
                    }
                    for edge in sorted(incoming, key=lambda e: e.source_id)
                ]
                # Same rule as live execution, over the history preceding this event.
                draft = _draft_source(
                    node_id=str(execute_id),
                    role_id=str(role_id),
                    primary_role_id=self._primary_role_id,
                    nodes=nodes,
                    output_id=output_id,
                    history=self._history[: index - 1],
                    inbound={edge.source_id for edge in incoming},
                )
                if draft is not None:
                    messages_list.insert(0, _draft_message(nodes[draft], str(execute_id)))
                messages = tuple(messages_list)
                executions += 1
                request = NodeExecutionRequest(
                    self.runtime_id,
                    execute_id,
                    self._roles[str(role_id)],
                    self.task_prompt,
                    tuple((skill_id, self._skills[skill_id]) for skill_id in skill_ids),
                    messages,
                    None if prev_executed_node is None else prev_executed_node.output,
                    executions,
                    int(
                        stable_hash(
                            {
                                "seed": self.seed,
                                "execution_index": executions,
                                "node_id": execute_id,
                            }
                        ).split(":", 1)[-1][:16],
                        16,
                    ),
                )
                raw_result = observation.get("node_result")
                if not isinstance(raw_result, dict) or set(raw_result) != {
                    "output",
                    "usage",
                    "metadata",
                }:
                    raise ValueError("history lacks an exact node execution result")
                result = NodeExecutionResult(
                    raw_result["output"],
                    BudgetVector.from_value(raw_result["usage"]),
                    raw_result["metadata"],
                )
                if not result.usage.fits_within(request.role.model_maximum):
                    raise ValueError("history node usage exceeds its reservation")
                usage = usage.add(result.usage)
                node = NodeState(
                    execute_id,
                    str(role_id),
                    skill_ids,
                    result.output,
                    1 if prev_executed_node is None else prev_executed_node.output_version + 1,
                )
                expected_observation.update(
                    {
                        "node_request": request.to_value(),
                        "node_result": result.to_value(),
                        "node_id": execute_id,
                        "output_version": node.output_version,
                        "output_changed": (
                            prev_executed_node is not None
                            and prev_executed_node.output != result.output
                        ),
                        "reservation_id": f"{self.runtime_id}:graph:{index}:node:{executions}",
                    }
                )
                entry = ledger_entries.get(str(expected_observation["reservation_id"]))
                if (
                    entry is None
                    or entry.settlement is None
                    or entry.reservation.maximum != request.role.model_maximum
                    or entry.settlement.actual != result.usage
                ):
                    raise ValueError("saved node receipt disagrees with exact budget ledger entry")
                nodes[execute_id] = node
            if extra_edge is not None:
                edges.append(extra_edge)
            if expected_observation != observation:
                raise ValueError("saved observation does not match its declared graph effect")
        graph = GraphState(
            tuple(nodes[key] for key in sorted(nodes)),
            tuple(sorted(edges, key=lambda edge: (edge.source_id, edge.target_id))),
            output_id,
        )
        if (
            graph != self.graph
            or next_id != self._next_node_index
            or executions != self._executions
            or usage != self._node_usage
        ):
            raise ValueError("snapshot graph/counters disagree with the complete history")


__all__ = [
    "DRAFT_MESSAGE_PROTOCOL",
    "SNAPSHOT_VERSION",
    "ExecutionReceipt",
    "GraphRuntime",
    "IllegalGraphActionError",
    "RuntimePoisonedError",
    "TerminalEvaluator",
]
