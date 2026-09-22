from __future__ import annotations

import ast
import tomllib
from dataclasses import fields
from pathlib import Path

from skillev.environments.debug_oracle import default_debug_oracle
from skillev.environments.public import (
    PublicSocialRow,
    PublicSocialTask,
    PublicSystemIdentificationObservations,
    PublicSystemIdentificationTask,
)

ROOT = Path(__file__).resolve().parents[2]


def _import_targets(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    targets: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            targets.append(("." * node.level) + (node.module or ""))
    return tuple(targets)


def test_runtime_facing_modules_do_not_import_private_implementations() -> None:
    public_directories = (
        ROOT / "src/skillev/environments",
        ROOT / "src/skillev/evaluation",
    )
    for directory in public_directories:
        for path in directory.glob("*.py"):
            assert all("private" not in target.split(".") for target in _import_targets(path))


def test_model_facing_wheel_physically_excludes_private_implementation() -> None:
    model_source = ROOT / "src/skillev"
    private_source = ROOT / "packages/private-evaluation/src/skillev_private"
    assert private_source.is_dir()
    assert not any(
        "private" in path.relative_to(model_source).parts for path in model_source.rglob("*.py")
    )

    root_config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    private_config = tomllib.loads(
        (ROOT / "packages/private-evaluation/pyproject.toml").read_text(encoding="utf-8")
    )
    assert root_config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["src/skillev"]
    assert private_config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "src/skillev_private"
    ]


def test_public_dtos_have_no_protected_state_fields() -> None:
    public_types = (
        PublicSocialRow,
        PublicSocialTask,
        PublicSystemIdentificationObservations,
        PublicSystemIdentificationTask,
    )
    forbidden_fragments = ("answer", "correct", "latent", "seed", "truth")
    for public_type in public_types:
        names = (field.name.lower() for field in fields(public_type))
        assert all(fragment not in name for name in names for fragment in forbidden_fragments)


def test_public_default_oracle_is_explicitly_debug_only_and_deterministic() -> None:
    oracle = default_debug_oracle()
    assignment = tuple(
        (operator.operator_id, int(operator.operator_id == "contextual"))
        for operator in oracle.operators
    )
    first = oracle.draw(assignment=assignment, context="social", seed=11)
    replay = oracle.draw(assignment=assignment, context="social", seed=11)
    physical = oracle.draw(assignment=assignment, context="physical", seed=11)

    assert oracle.formal_evaluation is False
    assert first == replay
    assert 0 <= first.terminal_reward <= 1
    assert first.value("quality") > physical.value("quality")
