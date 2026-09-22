from __future__ import annotations

import ast
import inspect
from dataclasses import fields
from pathlib import Path

import skillev.audit as audit
import skillev.evolution as evolution
import skillev.runtime as runtime
from skillev.audit.complete_method import audit_full_shaped_attempt
from skillev.audit.no_bayesian import audit_no_bayesian_attempt
from skillev.audit.no_bayesian_source_reducer import FlowOnlyAuditSourceReducer
from skillev.audit.source_reducer import AuditSourceReducer
from skillev.evolution import AuthoringEdgeEvidence
from skillev.runtime import AttemptSucceeded, AttemptSupervisor, RetrievalInclusionReason
from skillev.runtime.attempt_publication import PublishedSuccessfulAttemptBundle

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "skillev"
PRIVATE_SRC = ROOT / "packages" / "private-evaluation" / "src" / "skillev_private"


def _python_files(path: Path) -> tuple[Path, ...]:
    return (path,) if path.is_file() else tuple(sorted(path.rglob("*.py")))


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _imports(path: Path) -> tuple[str, ...]:
    names: list[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.append(node.module)
    return tuple(names)


def test_full_method_packages_do_not_import_arm_types() -> None:
    violations: list[str] = []
    for root in (SRC / "contracts", SRC / "evolution"):
        for path in _python_files(root):
            source = path.read_text(encoding="utf-8")
            if "FlowOnly" in source:
                violations.append(f"{path.relative_to(ROOT)}: FlowOnly")
            for imported in _imports(path):
                if imported.startswith("skillev.experiments"):
                    violations.append(f"{path.relative_to(ROOT)}: {imported}")

    assert not violations, violations
    assert "FlowOnlyRetainEvidence" not in runtime.__all__
    assert "FlowOnlyRetainEvidence" not in evolution.__all__


def test_active_method_has_no_compensating_runtime_calls() -> None:
    forbidden = {
        "abort",
        "fail_attempt_with_unknown_usage",
        "reconcile_failed_attempt",
        "suppress",
    }
    roots = (
        SRC / "rollout",
        SRC / "runtime" / "bounded_agent.py",
        SRC / "training",
        SRC / "evolution",
    )
    violations: list[str] = []
    for root in roots:
        for path in _python_files(root):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and (name := _call_name(node)) in forbidden:
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}:{name}")

    assert not violations, violations
    # Resource semaphores may return capacity. The forbidden operation is
    # compensating unknown model usage, not any method named "release".
    from skillev.runtime import BudgetLedger

    assert not hasattr(BudgetLedger, "release")


def test_broad_resource_cleanup_cannot_swallow_an_unknown_outcome() -> None:
    violations: list[str] = []
    for path in _python_files(SRC):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            caught = node.type
            is_broad = caught is None or (
                isinstance(caught, ast.Name) and caught.id in {"BaseException", "Exception"}
            )
            if not is_broad or path == SRC / "runtime" / "attempt_worker.py":
                continue
            # Pool rollback / lease cleanup must preserve the original failure.
            # No return, replacement exception or successful fallback is allowed.
            last = node.body[-1]
            if (
                not isinstance(last, ast.Raise)
                or last.exc is not None
                or any(isinstance(child, ast.Return) for child in ast.walk(node))
            ):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")

    assert not violations, violations


def test_live_packages_do_not_import_offline_audit_or_maintenance() -> None:
    violations: list[str] = []
    for package in ("application.py", "rollout", "training", "evolution"):
        for path in _python_files(SRC / package):
            for imported in _imports(path):
                if imported.startswith(("skillev.audit", "skillev.maintenance")):
                    violations.append(f"{path.relative_to(ROOT)}: {imported}")

    assert not violations, violations


def test_active_rollout_has_no_environment_checkpoint_calls() -> None:
    violations: list[str] = []
    for path in _python_files(SRC / "rollout"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _call_name(node) in {"checkpoint", "restore"}:
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")

    assert not violations, violations


def test_public_surfaces_expose_only_full_skill_context_and_real_author() -> None:
    retired = {"RetrievedSkillContext", "SkillDisclosure", "ScriptedSkillAuthor"}
    assert retired.isdisjoint(runtime.__all__)
    assert retired.isdisjoint(evolution.__all__)
    assert "FullRetrievedSkillContext" in runtime.__all__


def test_retired_recovery_and_compatibility_symbols_are_absent() -> None:
    retired = {
        "NoOpenPhaseState",
        "OpenPhaseRuntimeState",
        "PhaseRuntimeState",
        "RetrievalPassage",
        "backend_request_id",
        "phase_state_from_value",
    }
    violations: list[str] = []
    for path in (*_python_files(SRC), *_python_files(PRIVATE_SRC)):
        source = path.read_text(encoding="utf-8")
        for symbol in retired:
            if symbol in source:
                violations.append(f"{path.relative_to(ROOT)}:{symbol}")

    assert retired.isdisjoint(runtime.__all__)
    assert not violations, violations


def test_environment_protocol_has_no_replay_or_idempotency_surface() -> None:
    roots = (
        SRC / "benchmarks",
        SRC / "rollout",
        SRC / "runtime" / "bounded_agent.py",
        SRC / "runtime" / "execution.py",
        PRIVATE_SRC / "benchmarks",
    )
    violations: list[str] = []
    for root in roots:
        for path in _python_files(root):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    if node.name in {"checkpoint", "restore"}:
                        violations.append(f"{path.relative_to(ROOT)}:{node.lineno}:{node.name}")
                    for argument in (*node.args.args, *node.args.kwonlyargs):
                        if argument.arg == "idempotency_key":
                            violations.append(
                                f"{path.relative_to(ROOT)}:{argument.lineno}:idempotency_key"
                            )
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    if node.target.id == "idempotency_key":
                        violations.append(f"{path.relative_to(ROOT)}:{node.lineno}:idempotency_key")

    assert not violations, violations


def test_attempt_builder_and_supervisor_are_closed_production_controls() -> None:
    roots = (
        SRC / "runtime" / "attempt_builders.py",
        SRC / "runtime" / "attempt_supervisor.py",
        SRC / "runtime" / "attempt_worker.py",
        SRC / "experiments" / "exact_attempts.py",
    )
    violations: list[str] = []
    for path in roots:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import | ast.ImportFrom):
                imported = (
                    tuple(alias.name for alias in node.names)
                    if isinstance(node, ast.Import)
                    else (() if node.module is None else (node.module,))
                )
                if any(name == "importlib" or name.startswith("importlib.") for name in imported):
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}:importlib")
            elif isinstance(node, ast.Call) and _call_name(node) in {"getattr", "hasattr"}:
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}:{_call_name(node)}")

    assert tuple(field.name for field in fields(AttemptSupervisor)) == ("publisher",)
    assert not violations, violations


def test_authoring_evidence_cannot_persist_raw_model_or_reward_content() -> None:
    forbidden = {
        "action_arguments",
        "action_text",
        "native_payload",
        "observation_text",
        "query",
        "reasoning_text",
        "reward",
        "submission",
    }
    names = {field.name for field in fields(AuthoringEdgeEvidence)}
    assert names.isdisjoint(forbidden)


def test_public_audit_is_bundle_gated_without_caller_supplied_method_controls() -> None:
    signature = inspect.signature(audit.audit_published_attempt)
    annotation = signature.parameters["bundle"].annotation
    assert annotation in {
        PublishedSuccessfulAttemptBundle,
        "PublishedSuccessfulAttemptBundle",
    }
    assert tuple(signature.parameters) == ("bundle", "resources", "formal_run_ledger")
    assert not {
        "diagnostics_config",
        "calibration_config",
        "evolution_config",
        "arm",
    } & set(signature.parameters)


def test_typed_audit_and_source_reducers_accept_only_published_bundles() -> None:
    """Formal audit never accepts an arbitrary log path or event iterable.

    The manifest is the only source-set authority.  This protects the exact
    full/no-Bayesian parser split without adding a compatibility parser.
    """

    for function in (
        audit.audit_published_attempt,
        audit_full_shaped_attempt,
        audit_no_bayesian_attempt,
    ):
        parameters = inspect.signature(function).parameters
        assert "bundle" in parameters
        assert not {"path", "events", "event_log", "iterable"} & set(parameters)
        assert not {
            "diagnostics_config",
            "calibration_config",
            "evolution_config",
            "arm",
            "kernel",
        } & set(parameters)

    for reducer in (AuditSourceReducer, FlowOnlyAuditSourceReducer):
        parameters = inspect.signature(reducer.consume_published_bundle).parameters
        assert tuple(parameters) == ("self", "bundle")


def test_attempt_success_carries_the_final_training_artifact() -> None:
    """A success outcome cannot be published without the evaluated checkpoint."""

    outcome_fields = {field.name for field in fields(AttemptSucceeded)}
    published_fields = {field.name for field in fields(PublishedSuccessfulAttemptBundle)}
    assert "final_training_artifact" in outcome_fields
    assert "final_training_artifact" in published_fields


def test_active_method_source_has_no_retired_wire_controls() -> None:
    retired = {
        "PosteriorContextMode",
        "split_mode_requirement_id",
        "run_iterations",
        "phi_planning_budget",
        "event_log_sha256",
    }
    violations: list[str] = []
    for path in (*_python_files(SRC), *_python_files(PRIVATE_SRC)):
        source = path.read_text(encoding="utf-8")
        for symbol in retired:
            if symbol in source:
                violations.append(f"{path.relative_to(ROOT)}:{symbol}")
    assert not violations, violations


def test_private_formal_worker_cannot_route_through_public_completion_smoke() -> None:
    roots = (
        PRIVATE_SRC / "experiments" / "attempt_worker.py",
        PRIVATE_SRC / "experiments" / "attempt_supervisor.py",
        PRIVATE_SRC / "experiments" / "benchmark_attempt_builders.py",
    )
    violations: list[str] = []
    for path in roots:
        source = path.read_text(encoding="utf-8")
        if "skillev.experiments.exact_attempts" in source:
            violations.append(f"{path.relative_to(ROOT)}:public completion builder")
        if "importlib" in source:
            violations.append(f"{path.relative_to(ROOT)}:dynamic import")
    assert not violations, violations


def test_retrieval_has_one_active_inclusion_reason() -> None:
    assert tuple(RetrievalInclusionReason) == (RetrievalInclusionReason.APPLICABILITY_MATCH,)
