"""Original HumanEval population identity and candidate diagnostics."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from enum import StrEnum

from skillev_private.direct_reference.manifests import PopulationManifest

_HUMANEVAL_ID = re.compile(r"^HumanEval/(?:0|[1-9][0-9]*)$")


class CodeTestSuite(StrEnum):
    HUMANEVAL_ORIGINAL = "humaneval-original-v1"
    HUMANEVAL_PLUS = "humaneval-plus"
    MBPP_BASE_PLUS = "mbpp-base-plus-v0.2.0"


@dataclass(frozen=True, slots=True)
class PythonCandidateDiagnostics:
    parse_status: str
    syntax_valid: bool
    top_level_definition_count: int | None
    contains_markdown: bool


def validate_humaneval_manifest(
    manifest: PopulationManifest,
    *,
    expected_population_id: str = "humaneval-native-128-v13",
    expected_dataset_revision: str = "openai-humaneval-v1",
    expected_selection_rule: str = "frozen-native-ids-128",
) -> None:
    if manifest.population_id != expected_population_id:
        raise ValueError("HumanEval population differs")
    if manifest.dataset_revision != expected_dataset_revision:
        raise ValueError("HumanEval dataset revision differs")
    if manifest.selection_rule != expected_selection_rule:
        raise ValueError("HumanEval selection rule differs")
    for entry in manifest.entries:
        if _HUMANEVAL_ID.fullmatch(entry.source_identity) is None:
            raise ValueError(f"non-HumanEval source identity: {entry.source_identity}")
        if int(entry.source_identity.removeprefix("HumanEval/")) >= 164:
            raise ValueError("HumanEval source identity lies outside the original 164 tasks")


def diagnose_python_candidate(source: str | None) -> PythonCandidateDiagnostics:
    if source is None:
        return PythonCandidateDiagnostics("empty", False, None, False)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return PythonCandidateDiagnostics("extracted", False, None, "```" in source)
    definitions = sum(
        isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        for node in tree.body
    )
    return PythonCandidateDiagnostics("extracted", True, definitions, "```" in source)


__all__ = [
    "CodeTestSuite",
    "PythonCandidateDiagnostics",
    "diagnose_python_candidate",
    "validate_humaneval_manifest",
]
