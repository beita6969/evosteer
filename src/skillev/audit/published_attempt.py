"""Single bundle-gated audit dispatch for all seven preregistered arms."""

from __future__ import annotations

from typing import TypeAlias

from skillev.experiments import AttemptPurpose, FormalRunLedger
from skillev.runtime import AttemptBuilderKind, PublishedSuccessfulAttemptBundle

from .arm_kernels import audit_full_shaped_arm
from .complete_method import AuditResources, CompleteMethodAuditResult
from .no_bayesian import NoBayesianAuditResult, audit_no_bayesian_attempt
from .published_identity import read_published_attempt_identity

PublishedAttemptAuditResult: TypeAlias = CompleteMethodAuditResult | NoBayesianAuditResult


def audit_published_attempt(
    bundle: PublishedSuccessfulAttemptBundle,
    *,
    resources: AuditResources,
    formal_run_ledger: FormalRunLedger | None = None,
) -> PublishedAttemptAuditResult:
    exact = PublishedSuccessfulAttemptBundle.open_exact(bundle.directory)
    identity = read_published_attempt_identity(exact)
    if identity.purpose is AttemptPurpose.FORMAL_BENCHMARK_TRAINING:
        if not isinstance(formal_run_ledger, FormalRunLedger):
            raise ValueError("formal audit requires the manifest-selected terminal ledger")
        formal_run_ledger.require_bundle_terminal_success(exact)
    elif formal_run_ledger is not None:
        raise ValueError("correctness-fixture audit cannot carry a formal run ledger")
    if exact.builder_kind is AttemptBuilderKind.NO_BAYESIAN:
        return audit_no_bayesian_attempt(exact, resources=resources)
    return audit_full_shaped_arm(exact, resources=resources)


__all__ = ["PublishedAttemptAuditResult", "audit_published_attempt"]
