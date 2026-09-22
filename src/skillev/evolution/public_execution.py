"""Typed public execution snippets; never accepts an evaluator/native payload."""

import json
from dataclasses import dataclass

from skillev.contracts import JsonValue, TrajectoryStep, canonical_json


@dataclass(frozen=True, slots=True)
class PublicExecutionSnippet:
    action_text: str
    observation_text: str
    contrasts_json: tuple[str, ...] = ()

    @classmethod
    def from_step(cls, step: TrajectoryStep) -> "PublicExecutionSnippet":
        # These are precisely the two public wire fields. Do not serialize the
        # whole record, terminal reward, environment internals or reasoning.
        return cls(action_text=step.action_text, observation_text=step.observation_text)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "action_text": self.action_text,
            "observation_text": self.observation_text,
            **(
                {"same_source_contrasts": [json.loads(row) for row in self.contrasts_json]}
                if self.contrasts_json
                else {}
            ),
        }

    @classmethod
    def from_value(cls, value: object) -> "PublicExecutionSnippet":
        if not isinstance(value, dict):
            raise TypeError("public execution snippet must be an object")
        action, observation = value.get("action_text"), value.get("observation_text")
        if not isinstance(action, str) or not isinstance(observation, str):
            raise TypeError("public execution wire fields must be text")
        contrasts = value.get("same_source_contrasts", [])
        if not isinstance(contrasts, list) or any(
            not isinstance(row, dict)
            or row.get("format")
            not in {"same-source-observational-contrast@1", "same-source-observational-contrast@2"}
            for row in contrasts
        ):
            raise ValueError("public contrast evidence format differs")
        return cls(action, observation, tuple(canonical_json(row) for row in contrasts))
