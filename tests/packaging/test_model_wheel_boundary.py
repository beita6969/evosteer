from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from scripts.check_model_wheel import (
    check_dependency_light_public_imports,
    check_model_wheel,
)


def _wheel(path: Path, names: tuple[str, ...]) -> Path:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name in names:
            archive.writestr(name, "")
    return path


def test_model_wheel_accepts_public_runtime_files(tmp_path: Path) -> None:
    path = _wheel(
        tmp_path / "public.whl",
        ("skillev/runtime/execution.py", "skillev/environments/public.py"),
    )
    check_model_wheel(path)


@pytest.mark.parametrize(
    "retired_path",
    [
        "skillev/orchestration/features.py",
        "skillev/runtime/trajectory.py",
    ],
)
def test_model_wheel_rejects_retired_scientific_authorities(
    tmp_path: Path,
    retired_path: str,
) -> None:
    path = _wheel(tmp_path / "retired.whl", (retired_path,))

    with pytest.raises(ValueError):
        check_model_wheel(path)


@pytest.mark.parametrize(
    "private_path",
    [
        "skillev_private/evaluation/system_identification.py",
        "skillev/environments/private/social_dgp.py",
        "skillev/evaluation/private/exact_match.py",
    ],
)
def test_model_wheel_rejects_private_implementation(
    tmp_path: Path,
    private_path: str,
) -> None:
    path = _wheel(tmp_path / "private.whl", (private_path,))
    with pytest.raises(ValueError):
        check_model_wheel(path)


def test_model_wheel_rejects_an_import_of_private_evaluation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "private-import.whl"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "skillev/runtime/leak.py",
            "from skillev_private.evaluation import exact_match\n",
        )

    with pytest.raises(ValueError):
        check_model_wheel(path)


def test_dependency_light_public_imports_accept_a_wheel_without_policy_libraries(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dependency-light.whl"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("skillev/__init__.py", "")
        archive.writestr("skillev/scoring/__init__.py", "")
        archive.writestr("skillev/scoring/rendering.py", "")
        archive.writestr("skillev/rollout/__init__.py", "")
        archive.writestr("skillev/diagnostics/__init__.py", "")
        archive.writestr("skillev/calibration/__init__.py", "")
        archive.writestr("skillev/evolution/__init__.py", "")
        archive.writestr("skillev/benchmarks/__init__.py", "")
        archive.writestr("skillev/experiments/__init__.py", "")

    check_dependency_light_public_imports(path)


def test_dependency_light_public_imports_reject_a_heavy_scoring_import(
    tmp_path: Path,
) -> None:
    path = tmp_path / "heavy-scoring.whl"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("skillev/__init__.py", "")
        archive.writestr("skillev/scoring/__init__.py", "import torch\n")
        archive.writestr("skillev/scoring/rendering.py", "")
        archive.writestr("skillev/rollout/__init__.py", "")
        archive.writestr("skillev/diagnostics/__init__.py", "")
        archive.writestr("skillev/calibration/__init__.py", "")
        archive.writestr("skillev/evolution/__init__.py", "")

    with pytest.raises(ValueError):
        check_dependency_light_public_imports(path)


def test_dependency_light_public_imports_reject_a_heavy_rollout_import(
    tmp_path: Path,
) -> None:
    path = tmp_path / "heavy-rollout.whl"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("skillev/__init__.py", "")
        archive.writestr("skillev/scoring/__init__.py", "")
        archive.writestr("skillev/scoring/rendering.py", "")
        archive.writestr("skillev/rollout/__init__.py", "import transformers\n")
        archive.writestr("skillev/diagnostics/__init__.py", "")
        archive.writestr("skillev/calibration/__init__.py", "")
        archive.writestr("skillev/evolution/__init__.py", "")

    with pytest.raises(ValueError):
        check_dependency_light_public_imports(path)


def test_dependency_light_public_imports_reject_a_tokenizer_import(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tokenizer-scoring.whl"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("skillev/__init__.py", "")
        archive.writestr("skillev/scoring/__init__.py", "import tokenizers\n")
        archive.writestr("skillev/scoring/rendering.py", "")
        archive.writestr("skillev/rollout/__init__.py", "")
        archive.writestr("skillev/diagnostics/__init__.py", "")
        archive.writestr("skillev/calibration/__init__.py", "")
        archive.writestr("skillev/evolution/__init__.py", "")
        # Prove the runtime guard rejects the dependency rather than relying on
        # the isolated subprocess simply not having tokenizers installed.
        archive.writestr("tokenizers/__init__.py", "")

    with pytest.raises(ValueError):
        check_dependency_light_public_imports(path)


def test_dependency_light_public_imports_reject_a_heavy_diagnostics_import(
    tmp_path: Path,
) -> None:
    path = tmp_path / "heavy-diagnostics.whl"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("skillev/__init__.py", "")
        archive.writestr("skillev/scoring/__init__.py", "")
        archive.writestr("skillev/scoring/rendering.py", "")
        archive.writestr("skillev/rollout/__init__.py", "")
        archive.writestr("skillev/diagnostics/__init__.py", "import torch\n")
        archive.writestr("skillev/calibration/__init__.py", "")
        archive.writestr("skillev/evolution/__init__.py", "")

    with pytest.raises(ValueError):
        check_dependency_light_public_imports(path)


def test_dependency_light_public_imports_reject_a_heavy_calibration_import(
    tmp_path: Path,
) -> None:
    path = tmp_path / "heavy-calibration.whl"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("skillev/__init__.py", "")
        archive.writestr("skillev/scoring/__init__.py", "")
        archive.writestr("skillev/scoring/rendering.py", "")
        archive.writestr("skillev/rollout/__init__.py", "")
        archive.writestr("skillev/diagnostics/__init__.py", "")
        archive.writestr("skillev/calibration/__init__.py", "import peft\n")
        archive.writestr("skillev/evolution/__init__.py", "")
        # Ensure the import guard, rather than package availability, rejects it.
        archive.writestr("peft/__init__.py", "")

    with pytest.raises(ValueError):
        check_dependency_light_public_imports(path)


def test_dependency_light_public_imports_reject_a_heavy_evolution_import(
    tmp_path: Path,
) -> None:
    path = tmp_path / "heavy-evolution.whl"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("skillev/__init__.py", "")
        archive.writestr("skillev/scoring/__init__.py", "")
        archive.writestr("skillev/scoring/rendering.py", "")
        archive.writestr("skillev/rollout/__init__.py", "")
        archive.writestr("skillev/diagnostics/__init__.py", "")
        archive.writestr("skillev/calibration/__init__.py", "")
        archive.writestr("skillev/evolution/__init__.py", "import torch\n")

    with pytest.raises(ValueError):
        check_dependency_light_public_imports(path)
