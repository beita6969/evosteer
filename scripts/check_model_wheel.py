"""Verify that a model-facing SKILLEV wheel contains no private implementation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

FORBIDDEN_PREFIXES = (
    "skillev_private/",
    "skillev/environments/private/",
    "skillev/evaluation/private/",
)
FORBIDDEN_IMPORTS = (
    b"from skillev_private",
    b"import skillev_private",
)
RETIRED_SCIENTIFIC_PATHS = frozenset(
    {
        "skillev/orchestration/features.py",
        "skillev/runtime/trajectory.py",
    }
)

_DEPENDENCY_LIGHT_PUBLIC_IMPORTS = """
import sys

forbidden = {"torch", "transformers", "peft", "tokenizers"}

class BlockModelDependencies:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in forbidden:
            raise ImportError(f"model dependency import attempted: {fullname}")
        return None

sys.meta_path.insert(0, BlockModelDependencies())
import skillev.scoring
import skillev.scoring.rendering
import skillev.rollout
import skillev.diagnostics
import skillev.calibration
import skillev.evolution
import skillev.benchmarks
import skillev.experiments
loaded_roots = {name.partition(".")[0] for name in sys.modules}
if loaded_roots & forbidden:
    raise RuntimeError("dependency-light public imports loaded a model dependency")
"""


def check_model_wheel(path: Path) -> None:
    with ZipFile(path) as archive:
        forbidden = tuple(
            name
            for name in archive.namelist()
            if any(name.startswith(prefix) for prefix in FORBIDDEN_PREFIXES)
        )
        private_imports = tuple(
            name
            for name in archive.namelist()
            if name.endswith(".py")
            and any(fragment in archive.read(name) for fragment in FORBIDDEN_IMPORTS)
        )
        retired_scientific_paths = tuple(
            name for name in archive.namelist() if name in RETIRED_SCIENTIFIC_PATHS
        )
    if forbidden or private_imports or retired_scientific_paths:
        raise ValueError("Model-facing wheel contains private or retired implementation")


def check_dependency_light_public_imports(import_path: Path) -> None:
    """Import dependency-light public layers with model imports blocked."""

    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = str(import_path.resolve())
    with tempfile.TemporaryDirectory(prefix="skillev-public-import-") as working_directory:
        # The executable and program are fixed; only the isolated import path varies.
        completed = subprocess.run(  # noqa: S603
            [sys.executable, "-S", "-c", _DEPENDENCY_LIGHT_PUBLIC_IMPORTS],
            cwd=working_directory,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
    if completed.returncode != 0:
        raise ValueError("Model-facing public imports require a policy dependency")


def check_dependency_light_scoring_import(import_path: Path) -> None:
    """Backward-compatible name for the expanded public import gate."""

    check_dependency_light_public_imports(import_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    arguments = parser.parse_args()
    check_model_wheel(arguments.wheel)
    check_dependency_light_public_imports(arguments.wheel)


if __name__ == "__main__":
    main()
