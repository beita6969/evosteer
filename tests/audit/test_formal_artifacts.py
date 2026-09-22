from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from skillev.audit import AuditResources, FormalArtifactResolver, require_exact_formal_artifacts
from skillev.contracts import stable_hash
from skillev.experiments import ImplementationBuildIdentity
from skillev.policy import (
    ArtifactFileIdentity,
    BaseModelArtifactIdentity,
    PublicTokenizerKind,
    TokenizerArtifactIdentity,
)


@dataclass(frozen=True, slots=True)
class _FormalResolver(FormalArtifactResolver):
    base_model: BaseModelArtifactIdentity
    tokenizer: TokenizerArtifactIdentity
    implementation_build: ImplementationBuildIdentity

    def resolve_base_model(self, expected: BaseModelArtifactIdentity) -> BaseModelArtifactIdentity:
        del expected
        return self.base_model

    def resolve_tokenizer(self, expected: TokenizerArtifactIdentity) -> TokenizerArtifactIdentity:
        del expected
        return self.tokenizer

    def resolve_implementation_build(
        self, expected: ImplementationBuildIdentity
    ) -> ImplementationBuildIdentity:
        del expected
        return self.implementation_build


@dataclass(frozen=True, slots=True)
class _Artifacts:
    base_model: BaseModelArtifactIdentity
    tokenizer: TokenizerArtifactIdentity
    implementation: ImplementationBuildIdentity


def _artifacts() -> _Artifacts:
    def artifact_file(relative_path: str) -> ArtifactFileIdentity:
        return ArtifactFileIdentity(
            relative_path=relative_path,
            size_bytes=1,
            sha256=stable_hash({"formal-artifact-test": relative_path}),
        )

    return _Artifacts(
        base_model=BaseModelArtifactIdentity.create(
            backend_class="unit.Backend",
            upstream_revision="unit-revision",
            dtype_conversion_policy="unit-bf16",
            model_config=artifact_file("config.json"),
            generation_config=artifact_file("generation_config.json"),
            weight_index=None,
            weight_shards=(artifact_file("model.safetensors"),),
        ),
        tokenizer=TokenizerArtifactIdentity.create(
            kind=PublicTokenizerKind.QWEN,
            tokenizer_id="unit-tokenizer",
            revision="unit-revision",
            backend_serialization_hash=stable_hash("tokenizer-backend"),
            tokenizer_config_hash=stable_hash("tokenizer-config"),
            chat_template_hash=stable_hash("tokenizer-template"),
            special_tokens_hash=stable_hash("tokenizer-special"),
            added_tokens_hash=stable_hash("tokenizer-added"),
            transformers_version="unit-transformers",
            tokenizers_version="unit-tokenizers",
        ),
        implementation=ImplementationBuildIdentity(
            source_commit="1" * 40,
            source_tree_hash=stable_hash("source"),
            public_wheel_hash=stable_hash("public-wheel"),
            private_evaluation_wheel_hash=stable_hash("private-wheel"),
            lockfile_hash=stable_hash("lockfile"),
            python_version="3.11",
            torch_version="unit-torch",
            transformers_version="unit-transformers",
            peft_version="unit-peft",
            tokenizers_version="unit-tokenizers",
            deterministic_algorithms=True,
            cudnn_deterministic=True,
            cudnn_benchmark=False,
            matmul_allow_tf32=False,
        ),
    )


def test_formal_artifact_resolver_requires_all_three_exact_manifests() -> None:
    artifacts = _artifacts()

    require_exact_formal_artifacts(
        _FormalResolver(
            base_model=artifacts.base_model,
            tokenizer=artifacts.tokenizer,
            implementation_build=artifacts.implementation,
        ),
        base_model=artifacts.base_model,
        tokenizer=artifacts.tokenizer,
        implementation_build=artifacts.implementation,
    )

    with pytest.raises(ValueError):
        require_exact_formal_artifacts(
            _FormalResolver(
                base_model=artifacts.base_model,
                tokenizer=artifacts.tokenizer,
                implementation_build=replace(
                    artifacts.implementation,
                    source_tree_hash="sha256:" + "e" * 64,
                ),
            ),
            base_model=artifacts.base_model,
            tokenizer=artifacts.tokenizer,
            implementation_build=artifacts.implementation,
        )


def test_audit_resources_does_not_claim_live_formal_artifact_verification_by_default() -> None:
    assert AuditResources(tokenizer_resolver=object()).formal_artifact_resolver is None
