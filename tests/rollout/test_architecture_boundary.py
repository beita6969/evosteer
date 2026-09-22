from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

import skillev.runtime as runtime

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ROLLOUT_ROOT = REPOSITORY_ROOT / "src" / "skillev" / "rollout"
MODEL_LIBRARY_ROOTS = frozenset({"torch", "transformers", "peft", "tokenizers"})
RETIRED_SCIENTIFIC_MODULES = frozenset(
    {
        "skillev.orchestration.features",
        "skillev.runtime.models",
        "skillev.runtime.openai_provider",
        "skillev.runtime.trajectory",
    }
)
RETIRED_ACTIVE_SYMBOLS = frozenset(
    {
        "BackwardTokenScorer",
        "ModelProvider",
        "ModelResponse",
        "OpenAICompatibleProvider",
        "PendingTrajectoryStep",
        "action_token_logprobs",
        "backward_token_logprobs",
        "forward_token_logprobs",
        "score_trajectory",
    }
)
RETIRED_PUBLIC_RUNTIME_SYMBOLS = frozenset(
    {
        "ActionParser",
        "BackwardTokenScorer",
        "BoundedAgentLoopState",
        "PendingTrajectoryStep",
        "PromptBuilder",
        "TrajectoryStep",
    }
)


@dataclass(frozen=True, slots=True)
class BoundaryViolation:
    line: int
    description: str


def _is_type_checking_guard(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute)
        and isinstance(test.value, ast.Name)
        and test.value.id == "typing"
        and test.attr == "TYPE_CHECKING"
    )


def _call_name(function: ast.expr) -> str | None:
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


class _ActiveRolloutBoundaryVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[BoundaryViolation] = []

    def _record(self, node: ast.AST, description: str) -> None:
        self.violations.append(
            BoundaryViolation(
                line=getattr(node, "lineno", 0),
                description=description,
            )
        )

    def _check_identifier(self, node: ast.AST, identifier: str) -> None:
        normalized = identifier.casefold()
        if identifier in RETIRED_ACTIVE_SYMBOLS:
            self._record(node, f"retired active symbol {identifier}")
        if "optimizer" in normalized or normalized == "optim":
            self._record(node, f"optimizer symbol {identifier}")
        if identifier in MODEL_LIBRARY_ROOTS:
            self._record(node, f"model-library symbol {identifier}")
        if identifier == "BACKWARD_POLICY":
            self._record(node, "backward policy role")

    def visit_If(self, node: ast.If) -> None:
        if _is_type_checking_guard(node.test):
            # Type-only imports do not enter the active rollout graph.
            for statement in node.orelse:
                self.visit(statement)
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.partition(".")[0]
            if root in MODEL_LIBRARY_ROOTS:
                self._record(node, f"model-library import {alias.name}")
            if alias.name in RETIRED_SCIENTIFIC_MODULES:
                self._record(node, f"retired scientific module import {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        root = module.partition(".")[0]
        if root in MODEL_LIBRARY_ROOTS:
            self._record(node, f"model-library import {module}")
        if module in RETIRED_SCIENTIFIC_MODULES:
            self._record(node, f"retired scientific module import {module}")
        if module.startswith("skillev.policy") and module != "skillev.policy.interface":
            self._record(node, f"non-interface policy import {module}")
        for alias in node.names:
            self._check_identifier(node, alias.name)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        self._check_identifier(node, node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._check_identifier(node, node.attr)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if (
            isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
            and "logprob" in node.slice.value.casefold()
        ):
            self._record(node, "logprob mapping channel")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)
        if name == "backward":
            self._record(node, "backward call")
        if name is not None:
            self._check_identifier(node, name)
        self.generic_visit(node)


def _boundary_violations(source: str) -> tuple[BoundaryViolation, ...]:
    visitor = _ActiveRolloutBoundaryVisitor()
    visitor.visit(ast.parse(source))
    return tuple(visitor.violations)


@pytest.mark.parametrize(
    "source",
    [
        "from skillev.orchestration.features import ContextFeature",
        "from skillev.runtime.trajectory import TrajectoryStep",
        "from skillev.runtime import BackwardTokenScorer",
        "from skillev.runtime import ModelProvider",
        "response.action_token_logprobs",
        "score_trajectory(record)",
        "loss.backward()",
        "optimizer.step()",
        "import torch",
        "from transformers import AutoModel",
        "from peft import LoraConfig",
        "from tokenizers import Tokenizer",
        "from skillev.policy.hf_backbone import HFPolicyBackbone",
    ],
)
def test_architecture_boundary_detector_rejects_known_active_paths(source: str) -> None:
    assert _boundary_violations(source)


def test_architecture_boundary_detector_ignores_type_only_model_imports() -> None:
    source = """
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel
    from tokenizers import Tokenizer
"""

    assert _boundary_violations(source) == ()


def test_active_rollout_sources_do_not_cross_scientific_boundaries() -> None:
    modules = tuple(sorted(ROLLOUT_ROOT.rglob("*.py")))
    assert modules
    violations: list[str] = []

    for module in modules:
        source = module.read_text(encoding="utf-8")
        relative = module.relative_to(REPOSITORY_ROOT)
        violations.extend(
            f"{relative}:{violation.line}: {violation.description}"
            for violation in _boundary_violations(source)
        )

    assert not violations, "Active rollout architecture violations:\n" + "\n".join(violations)


def test_public_runtime_does_not_export_duplicate_scientific_authorities() -> None:
    public_exports = frozenset(runtime.__all__)

    assert public_exports.isdisjoint(RETIRED_PUBLIC_RUNTIME_SYMBOLS)
    assert all(not hasattr(runtime, symbol) for symbol in RETIRED_PUBLIC_RUNTIME_SYMBOLS)
