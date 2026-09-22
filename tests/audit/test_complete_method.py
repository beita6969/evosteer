from __future__ import annotations

import inspect
from dataclasses import dataclass

import pytest

import skillev.audit as audit
from skillev.audit.complete_method import audit_full_shaped_attempt
from skillev.audit.final_state import AuditedFinalTrainingState
from skillev.audit.tokenizer import require_exact_tokenizer
from skillev.contracts import (
    ContextFeature,
    FailureMode,
    HorizonBucket,
    PosteriorCellState,
    TokenBucket,
)
from skillev.experiments import TrainingEvolutionCounts
from skillev.policy import PublicTokenizerIdentity, PublicTokenizerKind
from skillev.runtime import SkillLibraryState


@dataclass(frozen=True, slots=True)
class _Tokenizer:
    tokenizer_id: str
    revision: str
    content_hash: str

    def encode(self, text: str) -> list[int]:
        return [len(text)]

    def encode_authoring_prompt(self, text: str) -> list[int]:
        return [len(text)]

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return "".join(chr(item) for item in token_ids)

    @property
    def public_identity(self) -> PublicTokenizerIdentity:
        return PublicTokenizerIdentity(
            kind=PublicTokenizerKind.QWEN,
            tokenizer_id=self.tokenizer_id,
            revision=self.revision,
            content_hash=self.content_hash,
        )


@dataclass(frozen=True, slots=True)
class _Resolver:
    tokenizer: _Tokenizer

    def resolve(self, identity: PublicTokenizerIdentity) -> _Tokenizer:
        del identity
        return self.tokenizer


def _identity() -> PublicTokenizerIdentity:
    return PublicTokenizerIdentity(
        kind=PublicTokenizerKind.QWEN,
        tokenizer_id="audit-tokenizer@1",
        revision="unit-test",
        content_hash="sha256:" + "a" * 64,
    )


def test_public_audit_has_one_bundle_gated_entrypoint() -> None:
    assert tuple(inspect.signature(audit.audit_published_attempt).parameters) == (
        "bundle",
        "resources",
        "formal_run_ledger",
    )
    assert not hasattr(audit, "audit_complete_method_run")


def test_full_shaped_audit_selects_its_kernel_from_the_published_identity() -> None:
    signature = inspect.signature(audit_full_shaped_attempt)
    assert tuple(signature.parameters) == ("bundle", "resources")
    assert not {
        "diagnostics_config",
        "calibration_config",
        "evolution_config",
        "arm",
        "kernel",
    } & set(signature.parameters)


def test_exact_tokenizer_resolver_rejects_another_pinned_artifact() -> None:
    identity = _identity()
    resolver = _Resolver(
        _Tokenizer(
            tokenizer_id=identity.tokenizer_id,
            revision=identity.revision,
            content_hash=identity.content_hash,
        )
    )
    assert require_exact_tokenizer(resolver, identity) == resolver.tokenizer

    with pytest.raises(ValueError):
        require_exact_tokenizer(
            _Resolver(
                _Tokenizer(
                    tokenizer_id=identity.tokenizer_id,
                    revision="other-revision",
                    content_hash=identity.content_hash,
                )
            ),
            identity,
        )


def test_final_state_keys_calibration_cells_by_skill_and_context() -> None:
    z = ContextFeature(
        context="audit-context",
        failure_mode=FailureMode.SUCCESS,
        token_bucket=TokenBucket.LE_1K,
        horizon_bucket=HorizonBucket.LE_3,
    )
    first = PosteriorCellState(
        skill_id="skill-alpha",
        z=z,
        alpha=1.0,
        beta_count=1.0,
    )
    second = PosteriorCellState(
        skill_id="skill-beta",
        z=z,
        alpha=1.0,
        beta_count=1.0,
    )

    state = AuditedFinalTrainingState(
        library=SkillLibraryState.from_seed_documents(()),
        calibration_cells=(first, second),
        final_policy_snapshot_id="policy@2",
        final_optimizer_step=2,
        evolution_counts=TrainingEvolutionCounts(
            training_step_count=2,
            phase_count=1,
            cycle_count=1,
            action_count=1,
        ),
    )

    assert state.calibration_cells == (first, second)
    with pytest.raises(ValueError):
        AuditedFinalTrainingState(
            library=SkillLibraryState.from_seed_documents(()),
            calibration_cells=(first, first),
            final_policy_snapshot_id="policy@2",
            final_optimizer_step=2,
            evolution_counts=TrainingEvolutionCounts(
                training_step_count=2,
                phase_count=1,
                cycle_count=1,
                action_count=1,
            ),
        )
