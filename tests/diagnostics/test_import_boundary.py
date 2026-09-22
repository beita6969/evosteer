from __future__ import annotations

import subprocess
import sys


def test_diagnostics_import_is_dependency_light() -> None:
    script = """
import sys
import skillev.diagnostics
loaded = {name.partition('.')[0] for name in sys.modules}
assert loaded.isdisjoint({'torch', 'transformers', 'peft', 'tokenizers'})
"""
    subprocess.run([sys.executable, "-c", script], check=True)  # noqa: S603
