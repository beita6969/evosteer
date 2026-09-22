"""Diagnose real local import routes, including aliased withdrawn selectors."""

from pathlib import Path

import pytest

from scripts.audit_step0_interventions import dependency_inventory


def test_dependency_inventory_follows_aliased_and_relative_imports_without_execution(tmp_path):
    package = tmp_path / "module"
    package.mkdir()
    (package / "__init__.py").write_text("")
    entry = tmp_path / "entry.py"
    entry.write_text("from module.route import dispatch\ndispatch()\n")
    (package / "route.py").write_text(
        "from .selector import choose_order_invariant_pairwise_candidate as choose\n"
        "def dispatch():\n    return choose(())\n"
        "raise RuntimeError('inventory must not execute this module')\n"
    )
    (package / "selector.py").write_text(
        "def choose_order_invariant_pairwise_candidate(value):\n    raise RuntimeError\n"
    )
    report = dependency_inventory((entry,), (tmp_path,))
    assert len(report["source_files"]) == 3
    assert len(report["source_calls"]) == 1
    assert report["source_calls"][0]["intervention"] == "choose_order_invariant_pairwise_candidate"
    with pytest.raises(ValueError):
        dependency_inventory((entry,), (tmp_path,), maximum_files=1)


def test_clean_cli_explicit_factory_actor_and_dispatch_have_no_selector_call_dependency():
    root = Path(__file__).resolve().parents[2]
    sources = (
        root / "scripts/run_step0_integrity_paired.py",
        root / "packages/private-evaluation/src/skillev_private/evaluation/integrity_runtime.py",
        root / "src/skillev/evaluation/integrity_actor.py",
        root / "src/skillev/evaluation/step0_dispatch.py",
    )
    report = dependency_inventory(
        sources, (root / "src", root / "packages/private-evaluation/src", root)
    )
    assert len(report["source_files"]) > len(sources)
    assert not {row["intervention"] for row in report["source_calls"]} & {
        "choose_order_invariant_pairwise_candidate",
        "choose_publicly_dominant_candidate",
    }
