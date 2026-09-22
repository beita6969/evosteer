"""Public, path-free identities for the trainable policy partition.

The frozen Qwen deployment is deliberately separate from the two LoRA adapters
and the Z head.  Formal ablations must start from exactly the same bytes in
that trainable partition; a deployment/configuration hash is not enough to
prove that property.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from skillev.contracts import JsonValue, normalize_json, stable_hash
from skillev.contracts.identity import validate_sha256

TRAINABLE_STATE_IDENTITY_FORMAT: Final = "skillev-trainable-state-identity@1"
PRIVATE_INITIAL_CHECKPOINT_BINDING_FORMAT: Final = "skillev-private-initial-checkpoint@1"


@dataclass(frozen=True, slots=True)
class TrainableStateIdentity:
    """Content identities for forward LoRA, backward LoRA, and Z.

    This value is safe to publish: it contains no checkpoint directory, model
    path, task material, or tensor payload.  The component hashes make a
    changed partition diagnosable while ``content_hash`` commits to their exact
    ordered tuple and the pinned deployment identity.
    """

    backbone_deployment_hash: str
    forward_adapter_hash: str
    backward_adapter_hash: str
    z_head_hash: str
    content_hash: str
    format: str = TRAINABLE_STATE_IDENTITY_FORMAT

    def __post_init__(self) -> None:
        for field in (
            "backbone_deployment_hash",
            "forward_adapter_hash",
            "backward_adapter_hash",
            "z_head_hash",
            "content_hash",
        ):
            validate_sha256(getattr(self, field))
        if self.format != TRAINABLE_STATE_IDENTITY_FORMAT:
            raise ValueError("unsupported trainable state identity format")
        expected = stable_hash(
            {
                "backbone_deployment_hash": self.backbone_deployment_hash,
                "backward_adapter_hash": self.backward_adapter_hash,
                "forward_adapter_hash": self.forward_adapter_hash,
                "z_head_hash": self.z_head_hash,
            }
        )
        if self.content_hash != expected:
            raise ValueError("trainable state content hash differs from components")

    @classmethod
    def create(
        cls,
        *,
        backbone_deployment_hash: str,
        forward_adapter_hash: str,
        backward_adapter_hash: str,
        z_head_hash: str,
    ) -> TrainableStateIdentity:
        return cls(
            backbone_deployment_hash=backbone_deployment_hash,
            forward_adapter_hash=forward_adapter_hash,
            backward_adapter_hash=backward_adapter_hash,
            z_head_hash=z_head_hash,
            content_hash=stable_hash(
                {
                    "backbone_deployment_hash": backbone_deployment_hash,
                    "backward_adapter_hash": backward_adapter_hash,
                    "forward_adapter_hash": forward_adapter_hash,
                    "z_head_hash": z_head_hash,
                }
            ),
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "backbone_deployment_hash": self.backbone_deployment_hash,
            "backward_adapter_hash": self.backward_adapter_hash,
            "content_hash": self.content_hash,
            "format": self.format,
            "forward_adapter_hash": self.forward_adapter_hash,
            "z_head_hash": self.z_head_hash,
        }

    @classmethod
    def from_value(cls, value: object) -> TrainableStateIdentity:
        normalized = normalize_json(value)
        fields = {
            "backbone_deployment_hash",
            "backward_adapter_hash",
            "content_hash",
            "format",
            "forward_adapter_hash",
            "z_head_hash",
        }
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("trainable state identity has incompatible fields")
        if any(type(normalized[field]) is not str for field in fields):
            raise TypeError("trainable state identity fields must be text")
        return cls(
            backbone_deployment_hash=normalized["backbone_deployment_hash"],
            forward_adapter_hash=normalized["forward_adapter_hash"],
            backward_adapter_hash=normalized["backward_adapter_hash"],
            z_head_hash=normalized["z_head_hash"],
            content_hash=normalized["content_hash"],
            format=normalized["format"],
        )


@dataclass(frozen=True, slots=True)
class PrivateInitialCheckpointBinding:
    """Private location of the one frozen initial trainable partition.

    The directory is an execution binding and intentionally never appears in
    a published identity.  Its content-level counterpart is carried by
    :class:`TrainableStateIdentity`, which every formal child must load and
    bind before its first rollout.
    """

    directory: str
    trainable_state: TrainableStateIdentity
    format: str = PRIVATE_INITIAL_CHECKPOINT_BINDING_FORMAT

    def __post_init__(self) -> None:
        if type(self.directory) is not str or not self.directory.strip():
            raise ValueError("initial checkpoint directory must be non-empty text")
        if not isinstance(self.trainable_state, TrainableStateIdentity):
            raise TypeError("initial checkpoint requires a trainable state identity")
        if self.format != PRIVATE_INITIAL_CHECKPOINT_BINDING_FORMAT:
            raise ValueError("unsupported private initial checkpoint binding format")

    @property
    def path(self) -> Path:
        """Resolve only at the private execution boundary."""

        return Path(self.directory)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "directory": self.directory,
            "format": self.format,
            "trainable_state": self.trainable_state.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> PrivateInitialCheckpointBinding:
        normalized = normalize_json(value)
        fields = {"directory", "format", "trainable_state"}
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("private initial checkpoint binding has incompatible fields")
        if type(normalized["directory"]) is not str or type(normalized["format"]) is not str:
            raise TypeError("private initial checkpoint binding text fields are invalid")
        return cls(
            directory=normalized["directory"],
            trainable_state=TrainableStateIdentity.from_value(normalized["trainable_state"]),
            format=normalized["format"],
        )


__all__ = [
    "PRIVATE_INITIAL_CHECKPOINT_BINDING_FORMAT",
    "TRAINABLE_STATE_IDENTITY_FORMAT",
    "PrivateInitialCheckpointBinding",
    "TrainableStateIdentity",
]
