#!/usr/bin/env python3
"""Enumerate intervention calls in explicit source files and counters in one private trace.

This is a diagnostic inventory, not an approval, attestation, or publication gate.
It never scans the private tree or computes file digests.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import yaml

from skillev.evaluation.sealed_candidates import CandidateReader
from skillev.evaluation.step0_integrity import InterventionCounts, load_integrity_arm

NAMES = {
    "_effective_native_actions",
    "_effective_alfworld_actions",
    "_reasoning_aligned_actions",
    "_webshop_catalog_navigation_action",
    "remove_evaluator_labels",
    "choose_order_invariant_pairwise_candidate",
    "choose_publicly_dominant_candidate",
    "retrieve_step_zero",
}


def inventory(paths: tuple[Path, ...]) -> list[dict[str, object]]:
    rows = []
    for path in paths:
        if path.stat().st_size > 512 * 1024:
            raise ValueError("inventory source exceeds the explicit file-size limit")
        tree = ast.parse(path.read_text())
        aliases = {
            alias.asname or alias.name: alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = (
                    aliases.get(node.func.id, node.func.id)
                    if isinstance(node.func, ast.Name)
                    else node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else ""
                )
                if name in NAMES:
                    rows.append(
                        {
                            "path": str(path),
                            "line": node.lineno,
                            "intervention": name,
                            "historical_module": path.name.startswith("legacy_"),
                        }
                    )
    return rows


def dependency_inventory(
    entrypoints: tuple[Path, ...], source_roots: tuple[Path, ...], *, maximum_files: int = 512
) -> dict[str, object]:
    """Follow explicit local imports without importing code or scanning directories.

    This is a conservative dependency inventory, not proof that every imported
    branch executes. Dynamic factories must be included as explicit entrypoints.
    """
    roots = tuple(path.resolve() for path in source_roots)
    pending = list(entrypoints)
    sources: set[Path] = set()
    edges = []
    while pending:
        path = pending.pop().resolve()
        if path in sources:
            continue
        if len(sources) >= maximum_files or path.stat().st_size > 512 * 1024:
            raise ValueError("dependency inventory exceeds its explicit resource limit")
        sources.add(path)
        root = next((root for root in roots if path.is_relative_to(root)), None)
        package = list(path.relative_to(root).with_suffix("").parts[:-1]) if root else []
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = package[: len(package) - node.level + 1] if node.level else []
                base = ".".join([*prefix, *([node.module] if node.module else [])])
                modules = [base, *(f"{base}.{alias.name}" for alias in node.names)]
            else:
                continue
            for module in modules:
                if not module or "*" in module:
                    continue
                relative = Path(*module.split("."))
                for source_root in roots:
                    matches = (
                        source_root / relative.with_suffix(".py"),
                        source_root / relative / "__init__.py",
                    )
                    target = next((candidate for candidate in matches if candidate.is_file()), None)
                    if target is not None:
                        edges.append(
                            {"source": str(path), "line": node.lineno, "target": str(target)}
                        )
                        pending.append(target)
                        break
    return {
        "scope": "conservative-local-import-dependencies; dynamic factories supplied explicitly",
        "source_files": [str(path) for path in sorted(sources)],
        "local_imports": edges,
        "source_calls": inventory(tuple(sorted(sources))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="*", type=Path)
    parser.add_argument(
        "--source-root",
        action="append",
        type=Path,
        default=[],
        help="Follow local imports in these explicit roots; no directory-tree search",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    report: dict[str, object] = {"source_calls": inventory(tuple(args.source))}
    if args.source_root:
        report["dependency_inventory"] = dependency_inventory(
            tuple(args.source), tuple(args.source_root)
        )
    if args.config is not None:
        with args.config.open() as stream:
            config = yaml.safe_load(stream)
        report["declared_arms"] = config.get("arms") or [load_integrity_arm(args.config).to_value()]
    if args.journal is not None:
        if not args.run_id:
            parser.error("--journal requires one explicit --run-id")
        reader = CandidateReader(args.journal)
        totals: dict[str, dict[str, int]] = {}
        try:
            for arm, payload in reader.connection.execute(
                "SELECT arm_id, payload FROM candidates WHERE run_id=?", (args.run_id,)
            ):
                row = json.loads(payload)
                counts = totals.setdefault(
                    arm, dict.fromkeys(InterventionCounts.__dataclass_fields__, 0)
                )
                for name, value in row.get("intervention_counts", {}).items():
                    counts[name] += value
        finally:
            reader.close()
        report["observed_candidate_counters"] = totals
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
