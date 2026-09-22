"""Canonical, private teacher-forcing inputs reused for scheduling and scoring."""

from __future__ import annotations

from dataclasses import dataclass

from skillev.contracts import TokenizerProtocol, TrajectoryRecord
from skillev.policy.interface import AdapterRole, ModelInputWindow, encode_policy_prompt

from .rendering import render_forward_prefix, render_hindsight_prefix


@dataclass(frozen=True, slots=True)
class PreparedEdge:
    step_index: int
    role: AdapterRole
    prefix_ids: tuple[int, ...]
    action_ids: tuple[int, ...]

    @property
    def full_length(self) -> int:
        return len(self.prefix_ids) + len(self.action_ids)


@dataclass(frozen=True, slots=True)
class PreparedEdgePlan:
    record: TrajectoryRecord
    initial_text: str
    tokenizer_id: str
    edges: tuple[PreparedEdge, ...]

    @property
    def token_cost(self) -> int:
        return max(1, sum(edge.full_length for edge in self.edges))

    def edge(self, step: int, role: AdapterRole) -> PreparedEdge:
        offset = 0 if role is AdapterRole.FORWARD_POLICY else self.record.horizon
        return self.edges[offset + step - 1]

    def wire_edges(self) -> list[dict[str, object]]:
        """Private same-host worker inputs; no re-encoding of sampled actions."""
        return [
            {"step": e.step_index, "role": e.role.value, "prefix": list(e.prefix_ids)}
            for e in self.edges
        ]

    @classmethod
    def from_wire_edges(
        cls, value: object, *, record: TrajectoryRecord, initial_text: str, tokenizer_id: str
    ) -> PreparedEdgePlan:
        if not isinstance(value, list) or len(value) != 2 * record.horizon:
            raise ValueError("prepared worker edges do not cover the complete trajectory")
        edges = []
        for i, row in enumerate(value):
            step = i % record.horizon + 1
            role = AdapterRole.FORWARD_POLICY if i < record.horizon else AdapterRole.BACKWARD_POLICY
            if (
                not isinstance(row, dict)
                or row.get("step") != step
                or row.get("role") != role.value
            ):
                raise ValueError("prepared worker edge order differs")
            prefix = row.get("prefix")
            if (
                not isinstance(prefix, list)
                or not prefix
                or any(type(t) is not int or t < 0 for t in prefix)
            ):
                raise ValueError("prepared worker prefix token IDs are invalid")
            edges.append(
                PreparedEdge(step, role, tuple(prefix), record.steps[step - 1].action_token_ids)
            )
        return cls(record, initial_text, tokenizer_id, tuple(edges))


def prepare_edge_plan(
    tokenizer: TokenizerProtocol,
    record: TrajectoryRecord,
    initial_text: str,
) -> PreparedEdgePlan:
    if tokenizer.tokenizer_id != record.tokenizer_id:
        raise ValueError("edge plan tokenizer differs from the recorded tokenizer")
    edges = []
    for role, render in (
        (AdapterRole.FORWARD_POLICY, render_forward_prefix),
        (AdapterRole.BACKWARD_POLICY, render_hindsight_prefix),
    ):
        for index, step in enumerate(record.steps, 1):
            rendered = render(initial_text, record.steps, index)
            expected = (
                step.forward_prefix_hash
                if role is AdapterRole.FORWARD_POLICY
                else step.hindsight_prefix_hash
            )
            # Preserve the existing recorded-prefix check; no new identity scheme.
            if rendered.prefix_hash != expected:
                raise ValueError("canonical scoring prefix differs from the recorded prefix")
            edges.append(
                PreparedEdge(
                    index,
                    role,
                    encode_policy_prompt(
                        tokenizer,
                        rendered.text,
                        initial_text=initial_text,
                        window=ModelInputWindow.from_meta(record.initial_context.meta),
                    ).ids,
                    step.action_token_ids,
                )
            )
    return PreparedEdgePlan(record, initial_text, tokenizer.tokenizer_id, tuple(edges))
