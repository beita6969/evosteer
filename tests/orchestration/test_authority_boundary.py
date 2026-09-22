from __future__ import annotations

import ast
from pathlib import Path

import pytest

import skillev.contracts as contracts
import skillev.orchestration as orchestration
import skillev.runtime as runtime
from skillev.contracts import ttb_calibration

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src" / "skillev"
CANONICAL_CALIBRATION_MODULE = SOURCE_ROOT / "contracts" / "ttb_calibration.py"
LEGACY_MODULES = frozenset(
    {
        "skillev.orchestration.features",
        "skillev.runtime.trajectory",
    }
)
CANONICAL_BAYESIAN_TYPES = frozenset(
    {
        "ContextFeature",
        "FailureMode",
        "HorizonBucket",
        "TokenBucket",
    }
)
RETIRED_ORCHESTRATION_EXPORTS = frozenset(
    {
        "BucketThresholds",
        "FailureMode",
        "FeatureConfig",
        "InvocationContext",
        "SkillContextFeatures",
        "build_context_features",
        "failure_mode_from_status",
    }
)


def _resolved_from_targets(node: ast.ImportFrom, module_path: Path | None) -> tuple[str, ...]:
    module = node.module or ""
    if node.level == 0:
        prefix = module
    elif module_path is None:
        return ()
    else:
        relative = module_path.relative_to(SOURCE_ROOT)
        package_parts = ["skillev", *relative.parent.parts]
        parent_hops = node.level - 1
        if parent_hops >= len(package_parts):
            return ()
        base_parts = package_parts[: len(package_parts) - parent_hops]
        prefix = ".".join((*base_parts, *module.split("."))) if module else ".".join(base_parts)

    targets = [prefix] if module else []
    targets.extend(f"{prefix}.{alias.name}" for alias in node.names)
    return tuple(targets)


def _legacy_imports(
    source: str,
    module_path: Path | None = None,
) -> tuple[tuple[int, str], ...]:
    imports: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports.extend(
                (node.lineno, alias.name) for alias in node.names if alias.name in LEGACY_MODULES
            )
        elif isinstance(node, ast.ImportFrom):
            imports.extend(
                (node.lineno, target)
                for target in _resolved_from_targets(node, module_path)
                if target in LEGACY_MODULES
            )
    return tuple(imports)


@pytest.mark.parametrize(
    "source",
    [
        "import skillev.orchestration.features",
        "from skillev.orchestration.features import FailureMode",
        "from skillev.orchestration import features",
        "import skillev.runtime.trajectory",
        "from skillev.runtime.trajectory import TrajectoryStep",
        "from skillev.runtime import trajectory",
    ],
)
def test_legacy_import_scanner_rejects_all_static_import_forms(source: str) -> None:
    assert _legacy_imports(source)


@pytest.mark.parametrize(
    ("module_path", "source"),
    [
        (SOURCE_ROOT / "orchestration" / "consumer.py", "from . import features"),
        (SOURCE_ROOT / "orchestration" / "consumer.py", "from .features import FailureMode"),
        (SOURCE_ROOT / "runtime" / "consumer.py", "from . import trajectory"),
        (SOURCE_ROOT / "runtime" / "consumer.py", "from .trajectory import TrajectoryStep"),
    ],
)
def test_legacy_import_scanner_resolves_relative_imports(
    module_path: Path,
    source: str,
) -> None:
    assert _legacy_imports(source, module_path)


def test_production_sources_do_not_import_retired_scientific_modules() -> None:
    violations: list[str] = []
    for module_path in sorted(SOURCE_ROOT.rglob("*.py")):
        relative_path = module_path.relative_to(REPOSITORY_ROOT)
        violations.extend(
            f"{relative_path}:{line}: import {module}"
            for line, module in _legacy_imports(
                module_path.read_text(encoding="utf-8"),
                module_path,
            )
        )

    assert not violations, "Retired scientific module imports:\n" + "\n".join(violations)


def test_retired_scientific_module_files_are_absent() -> None:
    assert not (SOURCE_ROOT / "orchestration" / "features.py").exists()
    assert not (SOURCE_ROOT / "runtime" / "trajectory.py").exists()


def test_phase_one_contracts_are_the_only_bayesian_type_definitions() -> None:
    definitions: dict[str, list[tuple[Path, int]]] = {
        type_name: [] for type_name in CANONICAL_BAYESIAN_TYPES
    }
    for module_path in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in definitions:
                definitions[node.name].append((module_path, node.lineno))

    for type_name in sorted(CANONICAL_BAYESIAN_TYPES):
        assert len(definitions[type_name]) == 1
        assert definitions[type_name][0][0] == CANONICAL_CALIBRATION_MODULE


def test_public_bayesian_authority_is_the_phase_one_contract_surface() -> None:
    assert CANONICAL_BAYESIAN_TYPES <= frozenset(contracts.__all__)
    for type_name in CANONICAL_BAYESIAN_TYPES:
        assert getattr(contracts, type_name) is getattr(ttb_calibration, type_name)

    assert frozenset(orchestration.__all__).isdisjoint(RETIRED_ORCHESTRATION_EXPORTS)
    assert all(not hasattr(orchestration, symbol) for symbol in RETIRED_ORCHESTRATION_EXPORTS)
    assert "TrajectoryStep" not in runtime.__all__
    assert not hasattr(runtime, "TrajectoryStep")
