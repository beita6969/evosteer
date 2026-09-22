"""Separate runtime-value, relative-flow and diagnostic outcome heads.

Runtime values consume public execution features only. Frozen encoder features
are used by the residual flow and diagnostic heads; final rewards are training labels,
never inference inputs. The paper specifies two diagnostic outcome heads but
not their widths or targets: this implementation explicitly chooses independent
128/GELU/sigmoid MLPs supervised by terminal-reward squared error. These heads
do not supply the runtime value or AnchorTB flow. Imports remain torch-lazy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module


def StateValueHeads(  # noqa: N802 - factory exposes the constructed module's public name
    *,
    encoding_dim: int,
    num_task_types: int = 12,
    feature_dim: int = 30,
    runtime_hidden_dim: int = 32,
    residual_hidden_dim: int = 128,
    outcome_hidden_dim: int = 128,
) -> Module:
    """Build an nn.Module; the residual output layer starts exactly at zero."""
    import torch
    from torch import nn

    dimensions = (
        encoding_dim,
        num_task_types,
        feature_dim,
        runtime_hidden_dim,
        residual_hidden_dim,
        outcome_hidden_dim,
    )
    if any(type(value) is not int or value < 1 for value in dimensions):
        raise ValueError("head dimensions must be positive integers")

    class _StateValueHeads(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoding_dim = encoding_dim
            self.num_task_types = num_task_types
            self.feature_dim = feature_dim
            self.runtime_head = nn.Sequential(
                nn.Linear(feature_dim + num_task_types, runtime_hidden_dim),
                nn.GELU(),
                nn.Linear(runtime_hidden_dim, 1),
            )
            residual_output = nn.Linear(residual_hidden_dim, 1)
            self.residual_head = nn.Sequential(
                nn.Linear(encoding_dim + feature_dim, residual_hidden_dim),
                nn.GELU(),
                residual_output,
            )
            nn.init.zeros_(residual_output.weight)
            nn.init.zeros_(residual_output.bias)
            self.outcome_heads = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(encoding_dim + feature_dim, outcome_hidden_dim),
                    nn.GELU(),
                    nn.Linear(outcome_hidden_dim, 1),
                    nn.Sigmoid(),
                )
                for _ in range(2)
            )

        def _features(self, features: Tensor) -> Tensor:
            if (
                features.ndim < 1
                or features.numel() == 0
                or features.shape[-1] != self.feature_dim
                or not features.is_floating_point()
                or not bool(torch.isfinite(features).all())
            ):
                raise ValueError("features must be finite public execution-feature vectors")
            return features.detach()

        def runtime_value(self, features: Tensor, task_types: Tensor) -> Tensor:
            features = self._features(features)
            if task_types.shape != features.shape[:-1] or task_types.dtype != torch.long:
                raise ValueError("one integer task-type index is required per feature vector")
            if bool(((task_types < 0) | (task_types >= self.num_task_types)).any()):
                raise ValueError("task-type index is outside the configured universe")
            one_hot = torch.nn.functional.one_hot(task_types, self.num_task_types).to(
                device=features.device, dtype=features.dtype
            )
            return cast(
                "Tensor",
                self.runtime_head(torch.cat((features, one_hot), dim=-1)).squeeze(-1).sigmoid(),
            )

        def _encoded_features(self, frozen_encoding: Tensor, features: Tensor) -> Tensor:
            features = self._features(features)
            if (
                frozen_encoding.shape != (*features.shape[:-1], self.encoding_dim)
                or not frozen_encoding.is_floating_point()
                or not bool(torch.isfinite(frozen_encoding).all())
            ):
                raise ValueError("frozen encoding must match the feature batch and encoding width")
            # The reference hidden state is unnormalized (per-dimension scale well
            # above the 30 features); at head lr 1e-3 the sigmoid outcome heads
            # saturated within three batches. Parameter-free RMS scaling keeps the
            # heads' inputs on a unit scale without changing what they can express.
            encoding = frozen_encoding.detach()
            encoding = encoding * torch.rsqrt(encoding.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            return torch.cat((encoding, features), dim=-1)

        def flow_residual(self, frozen_encoding: Tensor, features: Tensor) -> Tensor:
            return cast(
                "Tensor",
                self.residual_head(self._encoded_features(frozen_encoding, features)).squeeze(-1),
            )

        def outcome_values(self, frozen_encoding: Tensor, features: Tensor) -> Tensor:
            """Two independent diagnostic estimates; neither is an anchor or runtime feedback."""
            encoded = self._encoded_features(frozen_encoding, features)
            return torch.cat([head(encoded) for head in self.outcome_heads], dim=-1)

        def outcome_losses(
            self, frozen_encoding: Tensor, features: Tensor, rewards: Tensor
        ) -> Tensor:
            predictions = self.outcome_values(frozen_encoding, features)
            if (
                rewards.shape != predictions.shape[:-1]
                or not bool(torch.isfinite(rewards).all())
                or bool(((rewards < 0) | (rewards > 1)).any())
            ):
                raise ValueError("outcome reward labels must match states and lie in [0, 1]")
            errors = (predictions - rewards.detach().unsqueeze(-1)).square()
            return errors.reshape(-1, 2).mean(dim=0)

        def runtime_loss(self, features: Tensor, task_types: Tensor, rewards: Tensor) -> Tensor:
            prediction = self.runtime_value(features, task_types)
            if (
                rewards.shape != prediction.shape
                or not bool(torch.isfinite(rewards).all())
                or bool(((rewards < 0) | (rewards > 1)).any())
            ):
                raise ValueError("reference reward labels must match predictions and lie in [0, 1]")
            return (prediction - rewards.detach()).square().mean()

        def forward(self, features: Tensor, task_types: Tensor) -> Tensor:
            return self.runtime_value(features, task_types)

    return _StateValueHeads()
