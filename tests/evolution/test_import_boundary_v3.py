from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_evolution_import_is_dependency_light_and_does_not_load_audit() -> None:
    repository = Path(__file__).resolve().parents[2]
    script = f"""
import importlib.abc
import sys

blocked = {{"torch", "transformers", "peft", "tokenizers"}}

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in blocked:
            raise RuntimeError(fullname)
        return None

sys.meta_path.insert(0, Blocker())
sys.path.insert(0, {str(repository / "src")!r})
import skillev.evolution
assert "skillev.audit" not in sys.modules
"""
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-I", "-c", script],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
