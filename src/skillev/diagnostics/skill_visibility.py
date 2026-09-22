"""Actual admitted catalog/body visibility, never causal use or posterior credit."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from skillev.contracts import JsonValue, TrajectoryStep, canonical_json

if TYPE_CHECKING:
    from skillev.policy.interface import EncodedPolicyPrompt
    from skillev.rollout.generator import RolloutTokenizerProtocol
    from skillev.runtime import FullRetrievedSkillContext


def admitted_segments(
    prompt: EncodedPolicyPrompt, tokenizer: RolloutTokenizerProtocol
) -> tuple[str, ...] | None:
    """Do not join text across token ranges removed by the input window.

    Historical encoded prompts without range evidence are unknown on truncation.
    Complete returned envelopes, including escaped typed-history strings, must
    survive in one original contiguous range; skill mentions are insufficient.
    """
    lengths = prompt.retained_segment_lengths
    if lengths is None:
        return None if prompt.removed_tokens else (tokenizer.decode(prompt.ids),)
    if sum(lengths) != len(prompt.ids):
        raise ValueError("input evidence ranges differ from admitted input")
    offset = 0
    segments = []
    for size in lengths:
        if size:
            segments.append(tokenizer.decode(prompt.ids[offset : offset + size]))
        offset += size
    return tuple(segments)


@dataclass(frozen=True, slots=True)
class CatalogVisibility:
    entries: tuple[dict[str, JsonValue], ...]

    @classmethod
    def from_skills(cls, skills: tuple[FullRetrievedSkillContext, ...]) -> CatalogVisibility:
        from skillev.rollout.catalog import catalog_entry_value

        return cls(tuple(catalog_entry_value(skill) for skill in skills))

    def metrics(
        self, prompt: EncodedPolicyPrompt, tokenizer: RolloutTokenizerProtocol
    ) -> dict[str, JsonValue]:
        offered = [str(entry["skill_id"]) for entry in self.entries]
        visible: list[str] | None = offered
        if prompt.removed_tokens:
            segments = admitted_segments(prompt, tokenizer)
            seen: list[object] = []
            if segments is not None:
                for text in segments:
                    for match in re.finditer(r"(?m)^\[\d+\] (\{[^\n]*\})\s*$", text):
                        try:
                            seen.append(json.loads(match.group(1)))
                        except ValueError:
                            continue
            visible = (
                [str(entry["skill_id"]) for entry in self.entries if entry in seen]
                if segments is not None
                else None
            )
        return {
            "catalog_offered_skill_ids": cast(JsonValue, offered),
            "catalog_visible_skill_ids": cast(JsonValue, visible),
            "catalog_not_fully_visible_skill_ids": cast(
                JsonValue, [skill for skill in offered if skill not in visible]
            )
            if visible is not None
            else None,
            "catalog_visibility_basis": "admitted-contiguous-complete-entries@2"
            if prompt.removed_tokens
            else "untruncated-rendered-catalog",
        }

    def evidence(
        self,
        prompt: EncodedPolicyPrompt,
        tokenizer: RolloutTokenizerProtocol,
        *,
        previous_steps: tuple[TrajectoryStep, ...],
        library_version: str,
        step_index: int,
        phase: str,
    ) -> dict[str, JsonValue]:
        returned: list[tuple[dict[str, JsonValue], str]] = []
        for step in previous_steps:
            if not step.invoked_skill_ids or step.observation_status != "success":
                continue
            try:
                value = json.loads(step.observation_text)
            except ValueError:
                continue
            if not isinstance(value, dict) or not (
                value.get("status") == "skill-read"
                and value.get("skill_id") in step.invoked_skill_ids
                and value.get("library_version") == library_version
                and isinstance(value.get("version"), str)
                and isinstance(value.get("content"), str)
                and value["content"].strip()
            ):
                continue
            returned.append(
                (
                    {
                        "read_step_index": step.index,
                        "skill_id": value["skill_id"],
                        "skill_version": value["version"],
                        "library_version": library_version,
                    },
                    step.observation_text,
                )
            )
        segments = admitted_segments(prompt, tokenizer) if returned else ()
        visible: list[JsonValue] = []
        missing: list[JsonValue] = []
        if segments is not None:
            for ref, observation in returned:
                # Normal history contains the raw returned JSON. Typed history
                # represents that *whole* observation as a JSON string value.
                present = any(
                    observation in text or canonical_json(observation) in text for text in segments
                )
                (visible if present else missing).append(ref)
        return {
            "format": "skill-input-evidence@1",
            "step_index": step_index,
            "phase": phase,
            **self.metrics(prompt, tokenizer),
            "visible_skill_body_refs": visible if segments is not None else None,
            "not_fully_visible_skill_body_refs": missing if segments is not None else None,
            "body_visibility_basis": "complete-return-envelope-in-contiguous-admitted-tokens@1",
        }
