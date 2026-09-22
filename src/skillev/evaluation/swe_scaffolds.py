"""Pre-registered SWE-bench scaffold identities and formal-selection rules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml


class SWEScaffoldStatus(StrEnum):
    FORMAL_CANDIDATE = "formal-candidate"
    DIAGNOSTIC = "diagnostic"
    POSTHOC_DIAGNOSTIC = "posthoc-diagnostic"


@dataclass(frozen=True, slots=True)
class SWEScaffoldSpec:
    scaffold_id: str
    status: SWEScaffoldStatus
    prompt_profile: str
    tool_names: tuple[str, ...]
    editable: bool
    test_execution_allowed: bool
    syntax_feedback_allowed: bool
    horizon: int
    decoding_profile: str
    final_submission: str
    evidence_source: str
    frozen_before_final: bool

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (
                self.scaffold_id,
                self.prompt_profile,
                self.decoding_profile,
                self.final_submission,
                self.evidence_source,
            )
        ):
            raise ValueError("SWE scaffold identity fields must be non-empty")
        if self.horizon <= 0 or not self.tool_names:
            raise ValueError("SWE scaffold requires tools and a positive horizon")

    @property
    def permits_formal_comparison(self) -> bool:
        return self.status is SWEScaffoldStatus.FORMAL_CANDIDATE and self.frozen_before_final


def load_swe_scaffolds(path: Path) -> dict[str, SWEScaffoldSpec]:
    root = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(root, dict) or root.get("format") != "skillev-swe-scaffolds@1":
        raise ValueError("invalid SWE scaffold registry")
    rows = root.get("scaffolds")
    if not isinstance(rows, list):
        raise ValueError("SWE scaffold registry requires rows")
    output: dict[str, SWEScaffoldSpec] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("SWE scaffold row must be a mapping")
        spec = SWEScaffoldSpec(
            scaffold_id=str(row["id"]),
            status=SWEScaffoldStatus(str(row["status"])),
            prompt_profile=str(row["prompt_profile"]),
            tool_names=tuple(str(item) for item in row["tool_names"]),
            editable=bool(row["editable"]),
            test_execution_allowed=bool(row["test_execution_allowed"]),
            syntax_feedback_allowed=bool(row["syntax_feedback_allowed"]),
            horizon=int(row["horizon"]),
            decoding_profile=str(row["decoding_profile"]),
            final_submission=str(row["final_submission"]),
            evidence_source=str(row["evidence_source"]),
            frozen_before_final=bool(row["frozen_before_final"]),
        )
        if spec.scaffold_id in output:
            raise ValueError("duplicate SWE scaffold ID")
        output[spec.scaffold_id] = spec
    return output


__all__ = ["SWEScaffoldSpec", "SWEScaffoldStatus", "load_swe_scaffolds"]
