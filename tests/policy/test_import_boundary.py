from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts.check_model_wheel import check_dependency_light_public_imports

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
POLICY_ROOT = SOURCE_ROOT / "skillev" / "policy"
TRAINING_ROOT = SOURCE_ROOT / "skillev" / "training"
MODEL_DEPENDENCY_ROOTS = frozenset({"torch", "transformers", "peft", "tokenizers"})


@dataclass(frozen=True, slots=True)
class ModelImport:
    line: int
    module: str


def _is_type_checking_guard(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute)
        and isinstance(test.value, ast.Name)
        and test.value.id == "typing"
        and test.attr == "TYPE_CHECKING"
    )


class _RuntimeModelImportVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imports: list[ModelImport] = []

    def visit_If(self, node: ast.If) -> None:
        if _is_type_checking_guard(node.test):
            # The protected body is not executed at runtime; the else branch is.
            for statement in node.orelse:
                self.visit(statement)
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self._record(node, tuple(alias.name for alias in node.names))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is not None:
            self._record(node, (node.module,))

    def _record(self, node: ast.Import | ast.ImportFrom, modules: tuple[str, ...]) -> None:
        self.imports.extend(
            ModelImport(line=node.lineno, module=module)
            for module in modules
            if module.partition(".")[0] in MODEL_DEPENDENCY_ROOTS
        )


def _model_imports(source: str) -> tuple[ModelImport, ...]:
    visitor = _RuntimeModelImportVisitor()
    visitor.visit(ast.parse(source))
    return tuple(visitor.imports)


def _is_allowed_model_import(module_path: Path, model_import: ModelImport) -> bool:
    dependency_root = model_import.module.partition(".")[0]
    if module_path.is_relative_to(POLICY_ROOT):
        return True
    return dependency_root == "torch" and module_path.is_relative_to(TRAINING_ROOT)


def _disallowed_model_imports(
    module_path: Path,
    source: str,
) -> tuple[ModelImport, ...]:
    return tuple(
        model_import
        for model_import in _model_imports(source)
        if not _is_allowed_model_import(module_path, model_import)
    )


@pytest.mark.parametrize(
    ("source", "expected_modules"),
    [
        ("import torch", ("torch",)),
        ("import torch.nn.functional as functional", ("torch.nn.functional",)),
        (
            "import json, transformers.models.qwen2, peft",
            ("transformers.models.qwen2", "peft"),
        ),
        ("from transformers import AutoModel", ("transformers",)),
        ("from peft.tuners.lora import LoraLayer", ("peft.tuners.lora",)),
        ("from tokenizers.models import BPE", ("tokenizers.models",)),
        ("from skillev.contracts import stable_hash", ()),
    ],
)
def test_model_import_scanner_covers_import_forms(
    source: str, expected_modules: tuple[str, ...]
) -> None:
    assert tuple(item.module for item in _model_imports(source)) == expected_modules


def test_model_import_scanner_exempts_type_checking_body() -> None:
    source = """
if TYPE_CHECKING:
    import torch
    from transformers import AutoModel
if typing.TYPE_CHECKING:
    import peft.tuners.lora
"""

    assert _model_imports(source) == ()


@pytest.mark.parametrize(
    ("source", "expected_module"),
    [
        ("import torch", "torch"),
        ("if RUNTIME_CHECK:\n    import transformers", "transformers"),
        ("if not TYPE_CHECKING:\n    import peft", "peft"),
        (
            "if typing.TYPE_CHECKING:\n    import json\nelse:\n    import torch.nn",
            "torch.nn",
        ),
    ],
)
def test_model_import_scanner_keeps_runtime_and_non_guarded_imports(
    source: str, expected_module: str
) -> None:
    assert tuple(item.module for item in _model_imports(source)) == (expected_module,)


@pytest.mark.parametrize(
    ("module_path", "source", "expected_modules"),
    [
        (TRAINING_ROOT / "trainer.py", "import torch", ()),
        (TRAINING_ROOT / "trainer.py", "from torch.optim import AdamW", ()),
        (
            TRAINING_ROOT / "trainer.py",
            "from transformers import AutoModel",
            ("transformers",),
        ),
        (TRAINING_ROOT / "trainer.py", "import peft", ("peft",)),
        (TRAINING_ROOT / "trainer.py", "import tokenizers", ("tokenizers",)),
        (
            SOURCE_ROOT / "skillev" / "rollout" / "engine.py",
            "import torch",
            ("torch",),
        ),
        (
            SOURCE_ROOT / "skillev" / "rollout" / "engine.py",
            "from tokenizers import Tokenizer",
            ("tokenizers",),
        ),
        (
            POLICY_ROOT / "hf_backbone.py",
            "import torch, transformers, peft, tokenizers",
            (),
        ),
    ],
)
def test_model_import_allowlist_distinguishes_policy_and_training(
    module_path: Path,
    source: str,
    expected_modules: tuple[str, ...],
) -> None:
    assert (
        tuple(item.module for item in _disallowed_model_imports(module_path, source))
        == expected_modules
    )


def test_model_dependencies_follow_policy_and_training_runtime_boundary() -> None:
    violations: list[str] = []
    scanned_modules = tuple(sorted(SOURCE_ROOT.rglob("*.py")))

    assert scanned_modules, "src must contain Python modules"
    for module_path in scanned_modules:
        source = module_path.read_text(encoding="utf-8")
        relative_path = module_path.relative_to(REPOSITORY_ROOT)
        violations.extend(
            f"{relative_path}:{model_import.line}: import {model_import.module}"
            for model_import in _disallowed_model_imports(module_path, source)
        )

    assert not violations, (
        "torch runtime imports are allowed only under policy/training; "
        "transformers/peft/tokenizers only under policy:\n" + "\n".join(violations)
    )


def test_scoring_and_rollout_imports_do_not_load_or_require_model_dependencies() -> None:
    check_dependency_light_public_imports(SOURCE_ROOT)
