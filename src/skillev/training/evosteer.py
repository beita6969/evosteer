"""Synchronous mixed-source AnchorTB updates, independent of hindsight TTB.

The default streams one action's language-model graph at a time. A no-gradient
pass measures every action ratio, then the exact O(T) AnchorTB objective yields
distinct action and flow coefficients for immediate second-pass backpropagation.
Dense backpropagation remains available as a numerical reference.

Supervised states are pre-action states with a legal reference continuation:
every natural-reference state, and paired states after the forced ADD_AGENT.
The terminal state has no continuation and supplies only Eq. (7)'s boundary.
Value and diagnostic losses cannot affect actor/flow gradients or clipping.

Algorithm 1 step 5 fits the runtime value head. One joint AdamW step per batch
leaves it near its initial constant, so ``value_fit_steps - 1`` further
value-only passes follow the joint step, over this batch's legal reference
states plus a bounded window of earlier batches (``value_replay_batches``).
They move only the runtime value head; the window is checkpointed inside the
optimizer state dict.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from skillev.contracts.canonical import stable_hash
from skillev.contracts.evosteer import EvoTrajectory
from skillev.scoring.anchor_tb import anchor_tb_loss
from skillev.training.reference_statistics import ReferenceStatisticsSnapshot, StructureVisit


@dataclass(frozen=True, slots=True)
class EvoOptimizerConfig:
    actor_learning_rate: float = 5e-6
    head_learning_rate: float = 1e-3
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    beta: float = 1.0
    value_loss_weight: float = 1.0
    outcome_loss_weight: float = 1.0
    gradient_mode: str = "streaming"
    # Runtime value-head AdamW steps per batch, counting the joint step, so 1
    # means the joint step alone. More steps are needed for Algorithm 1
    # step 5 to actually fit v_hat instead of leaving it near its initial constant.
    value_fit_steps: int = 1
    # Number of recent updates, including the current one, whose legal reference
    # states make up the extra value passes' fit set. 1 fits only the current
    # batch and keeps no state between batches.
    value_replay_batches: int = 1
    # AdamW's first-moment decay per parameter group. The actor's gradient sits
    # at Adam's pure-noise floor, where a longer first moment leaves the directed
    # part of m_hat untouched (an EMA of a constant) and shrinks the noise part by
    # sqrt((1 - beta1) / (1 + beta1)); the far faster heads keep the torch default.
    # beta2 is the torch default for both groups.
    actor_beta1: float = 0.9
    head_beta1: float = 0.9

    def __post_init__(self) -> None:
        for name in ("actor_learning_rate", "head_learning_rate", "gradient_clip", "beta"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("actor_beta1", "head_beta1"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{name} must be finite and in [0, 1)")
        if any(
            not math.isfinite(v) or v < 0
            for v in (
                self.weight_decay,
                self.value_loss_weight,
                self.outcome_loss_weight,
            )
        ):
            raise ValueError("weight decay and supervised loss weights must be nonnegative")
        if self.gradient_mode not in {"streaming", "dense"}:
            raise ValueError("gradient_mode must be streaming or dense")
        for name in ("value_fit_steps", "value_replay_batches"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.value_replay_batches > 1 and self.value_fit_steps == 1:
            # Only the extra passes read the window; rejecting the combination
            # stops a config from keeping checkpoint state that is never used.
            raise ValueError("value_replay_batches > 1 requires value_fit_steps > 1")


# torch.optim.AdamW's own second-moment decay, named once because both parameter
# groups now state their betas explicitly to set beta1 apart.
ADAM_BETA2 = 0.999

# Stored in the optimizer state dict. The window is optimizer-side state of the
# value fit, alongside the head's AdamW moments.
VALUE_REPLAY_KEY = "evosteer_value_replay"
VALUE_REPLAY_FORMAT = "evosteer-value-replay@1"


@dataclass(frozen=True, slots=True, eq=False)
class _ValueReplayBatch:
    """One update's legal reference states: CPU tensors that are never mutated."""

    batch_id: str
    features: Any
    task_types: Any
    rewards: Any


def family_mean_loss(labels: list[tuple[int, float, int]]) -> float:
    """In-sample MSE of each state's per-family mean label.

    ``labels`` holds (task-type index, reward, state count) per trajectory. This
    is the best any family-only predictor can do on the same states, so the
    pre-fit ``value_loss`` should fall toward or below it once v_hat learns more
    than the family base rates.
    """
    total = sum(count for _, _, count in labels)
    if not total:
        return 0.0
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for family, reward, count in labels:
        sums[family] = sums.get(family, 0.0) + reward * count
        counts[family] = counts.get(family, 0) + count
    squared = sum(
        count * (reward - sums[family] / counts[family]) ** 2 for family, reward, count in labels
    )
    return squared / total


def structure_visit(state_json: str) -> StructureVisit:
    """Pool graph structure only; never include outputs, task text or runtime IDs."""
    state = json.loads(state_json)
    graph = state["graph"]
    nodes = sorted(graph["nodes"], key=lambda node: int(node["node_id"][1:]))
    indices = {node["node_id"]: i for i, node in enumerate(nodes)}
    structure = {
        "nodes": [{"role": node["role_id"], "skills": sorted(node["skill_ids"])} for node in nodes],
        "edges": sorted(
            (indices[edge["source_id"]], indices[edge["target_id"]], edge["protocol"])
            for edge in graph["edges"]
        ),
        "output": indices.get(graph["output_node_id"]),
    }
    return StructureVisit(stable_hash(structure), len(nodes), len(graph["edges"]), state["stopped"])


def _reference_state(trajectory: EvoTrajectory, index: int) -> bool:
    """A forced-action state has no unbiased rho continuation until after that action."""
    return trajectory.source == "natural_reference" or (
        trajectory.source in {"paired_treatment", "paired_control"} and index >= 1
    )


class EvoTrainer:
    def __init__(
        self,
        policy: Any,
        heads: Any,
        task_families: tuple[str, ...],
        config: EvoOptimizerConfig | None = None,
    ) -> None:
        import torch

        if not task_families or len(set(task_families)) != len(task_families):
            raise ValueError("task families must be a nonempty fixed unique universe")
        self.policy = policy
        self.heads = heads.to(device=policy.device, dtype=torch.float32)
        self.task_families = task_families
        self.config = config or EvoOptimizerConfig()
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": policy.trainable_parameters(),
                    "lr": self.config.actor_learning_rate,
                    "betas": (self.config.actor_beta1, ADAM_BETA2),
                },
                {
                    "params": heads.parameters(),
                    "lr": self.config.head_learning_rate,
                    "betas": (self.config.head_beta1, ADAM_BETA2),
                },
            ],
            weight_decay=self.config.weight_decay,
        )
        # Most recent earlier updates' reference states, oldest first; at most
        # value_replay_batches - 1 entries (empty when the window is 1).
        self._value_replay: tuple[_ValueReplayBatch, ...] = ()
        self._pending_value_replay: tuple[_ValueReplayBatch, ...] | None = None
        # The window lives in optimizer.state_dict(), so any checkpoint that
        # saves and restores the optimizer (the application saves
        # trainer.optimizer.state_dict() under its tensor checksum) carries it
        # atomically. A resumed run then refits on exactly the same window.
        self.optimizer.register_state_dict_post_hook(self._export_value_replay)
        self.optimizer.register_load_state_dict_pre_hook(self._stage_value_replay)
        self.optimizer.register_load_state_dict_post_hook(self._commit_value_replay)

    def apply_configured_hyperparameters(self) -> None:
        """Re-assert this run's AdamW hyperparameters over a restored state dict.

        torch's Optimizer.load_state_dict restores every parameter-group
        hyperparameter from the saved state, so a resumed run would otherwise
        silently keep the rates, decays and first moments its checkpoint was
        written at. Only the hyperparameters move: the moments and the step
        counts stay as saved, which is the point of resuming at all.
        """
        options = (
            {
                "lr": self.config.actor_learning_rate,
                "betas": (self.config.actor_beta1, ADAM_BETA2),
                "weight_decay": self.config.weight_decay,
            },
            {
                "lr": self.config.head_learning_rate,
                "betas": (self.config.head_beta1, ADAM_BETA2),
                "weight_decay": self.config.weight_decay,
            },
        )
        if len(self.optimizer.param_groups) != len(options):
            raise ValueError("restored optimizer has an unexpected parameter group count")
        for group, values in zip(self.optimizer.param_groups, options):
            group.update(values)

    def _export_value_replay(self, optimizer: Any, state_dict: dict[str, Any]) -> dict[str, Any]:
        if self.config.value_replay_batches > 1:
            state_dict[VALUE_REPLAY_KEY] = {
                "format": VALUE_REPLAY_FORMAT,
                "capacity": self.config.value_replay_batches,
                "batches": [
                    {
                        "batch_id": item.batch_id,
                        "features": item.features,
                        "task_types": item.task_types,
                        "rewards": item.rewards,
                    }
                    for item in self._value_replay
                ],
            }
        return state_dict

    def _stage_value_replay(self, optimizer: Any, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Validate the stored window; it is committed only if the whole load succeeds."""
        # torch passes a shallow copy, so popping leaves the caller's dict intact.
        stored = state_dict.pop(VALUE_REPLAY_KEY, None)
        keep = self.config.value_replay_batches - 1
        if not keep:
            if stored is not None:
                raise ValueError("optimizer state carries a value replay window this config lacks")
            self._pending_value_replay = ()
            return state_dict
        if (
            not isinstance(stored, Mapping)
            or set(stored) != {"format", "capacity", "batches"}
            or stored["format"] != VALUE_REPLAY_FORMAT
            or type(stored["capacity"]) is not int
            or stored["capacity"] != self.config.value_replay_batches
            or not isinstance(stored["batches"], list | tuple)
            or len(stored["batches"]) > keep
        ):
            raise ValueError("optimizer state lacks this config's value replay window")
        batches = tuple(self._replay_batch(item) for item in stored["batches"])
        if len({item.batch_id for item in batches}) != len(batches):
            raise ValueError("value replay window repeats a batch")
        self._pending_value_replay = batches
        return state_dict

    def _commit_value_replay(self, optimizer: Any) -> None:
        if self._pending_value_replay is not None:
            self._value_replay = self._pending_value_replay
            self._pending_value_replay = None

    def _replay_batch(self, value: Any) -> _ValueReplayBatch:
        import torch

        if not isinstance(value, Mapping) or set(value) != {
            "batch_id",
            "features",
            "task_types",
            "rewards",
        }:
            raise ValueError("malformed value replay batch")
        features, task_types, rewards = (value[k] for k in ("features", "task_types", "rewards"))
        if (
            not isinstance(value["batch_id"], str)
            or not value["batch_id"]
            or not all(isinstance(t, torch.Tensor) for t in (features, task_types, rewards))
            or features.dtype != torch.float32
            or features.ndim != 2
            or features.shape[1] != self.heads.feature_dim
            or task_types.dtype != torch.long
            or rewards.dtype != torch.float32
            or task_types.shape != features.shape[:1]
            or rewards.shape != features.shape[:1]
            or not bool(torch.isfinite(features).all())
            or not bool(torch.isfinite(rewards).all())
            or bool(((rewards < 0) | (rewards > 1)).any())
            or bool(((task_types < 0) | (task_types >= len(self.task_families))).any())
        ):
            raise ValueError("malformed value replay batch")
        return _ValueReplayBatch(
            value["batch_id"],
            features.detach().cpu(),
            task_types.detach().cpu(),
            rewards.detach().cpu(),
        )

    def _fit_value_head(
        self, current: _ValueReplayBatch, *, joint_step: bool
    ) -> dict[str, float | int]:
        """Algorithm 1 step 5: finish fitting v_hat after the joint step.

        The joint step already applied this batch's first value gradient. The
        remaining value_fit_steps - 1 passes are full-window gradient steps on
        the same optimizer, whose AdamW moments for the runtime head (lr
        head_learning_rate) belong to that head alone. AdamW skips parameters
        whose gradient is None, so every other parameter's gradient is set
        aside first. The actor, residual and diagnostic weights and their
        moments therefore cannot move, and their joint-step gradients are put
        back afterwards. The window is used whole and in a fixed order (older
        batches first, then trajectory order), with no sampling, so the fit
        is deterministic.
        """
        import torch

        window = (*self._value_replay, current)
        device = self.policy.device
        features = torch.cat([item.features for item in window]).to(device)
        task_types = torch.cat([item.task_types for item in window]).to(device)
        rewards = torch.cat([item.rewards for item in window]).to(device)
        count = int(rewards.numel())
        extra = self.config.value_fit_steps - 1 if count and self.config.value_loss_weight else 0
        value_parameters = tuple(self.heads.runtime_head.parameters())
        if extra:
            value_ids = {id(parameter) for parameter in value_parameters}
            held = [
                (parameter, parameter.grad)
                for group in self.optimizer.param_groups
                for parameter in group["params"]
                if id(parameter) not in value_ids and parameter.grad is not None
            ]
            for parameter, _ in held:
                parameter.grad = None
            try:
                for _ in range(extra):
                    for parameter in value_parameters:
                        parameter.grad = None
                    loss = self.heads.runtime_loss(features, task_types, rewards)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("nonfinite value-head fit loss")
                    (loss * self.config.value_loss_weight).backward()
                    for parameter in value_parameters:
                        if parameter.grad is not None and not bool(
                            torch.isfinite(parameter.grad).all()
                        ):
                            raise FloatingPointError("nonfinite value-head fit gradient")
                    # The same per-head norm limit as the joint step.
                    torch.nn.utils.clip_grad_norm_(
                        value_parameters, self.config.gradient_clip, error_if_nonfinite=True
                    )
                    self.optimizer.step()
            finally:
                for parameter, gradient in held:
                    parameter.grad = gradient
        fit_loss = 0.0
        if count:
            with torch.no_grad():
                predictions = self.heads.runtime_value(features, task_types)
                fit_loss = float((predictions - rewards).square().mean())
        keep = self.config.value_replay_batches - 1
        # Commit the window only after every pass has succeeded.
        self._value_replay = window[-keep:] if keep else ()
        return {
            "value_fit_steps": int(joint_step) + extra,
            "value_fit_states": count,
            "value_fit_loss": fit_loss,
        }

    def value(self, features: tuple[float, ...], family: str, *, heads: Any = None) -> float:
        import torch

        active = self.heads if heads is None else heads
        device = next(active.parameters()).device
        with torch.no_grad():
            result = active.runtime_value(
                torch.tensor([features], dtype=torch.float32, device=device),
                torch.tensor([self.task_families.index(family)], dtype=torch.long, device=device),
            )
        return float(result.item())

    def update(
        self,
        trajectories: tuple[EvoTrajectory, ...],
        snapshots: Mapping[str, ReferenceStatisticsSnapshot],
    ) -> dict[str, float | int]:
        import torch

        if not trajectories or len({x.sample_id for x in trajectories}) != len(trajectories):
            raise ValueError("training requires a complete batch of unique histories")
        if len({x.batch_id for x in trajectories}) != 1:
            raise ValueError("histories from different rollout batches cannot be mixed")
        # Validate the complete plan before changing any model/optimizer state.
        for trajectory in trajectories:
            if trajectory.reference_id != self.policy.reference_id:
                raise ValueError("trajectory was generated against another frozen reference")
            if trajectory.task.family not in self.task_families:
                raise ValueError("unknown task family")
            snapshot = snapshots[trajectory.statistics_context]
            if snapshot.context_version != trajectory.statistics_context:
                raise ValueError("anchor context does not match the trajectory")
            if snapshot.config.beta != self.config.beta:
                raise ValueError("anchor and objective reward temperatures differ")
        if any(item.batch_id == trajectories[0].batch_id for item in self._value_replay):
            # A replayed batch would enter the value fit window twice.
            raise ValueError("this rollout batch was already fitted")
        self.optimizer.zero_grad(set_to_none=True)
        actor_loss = 0.0
        action_count = 0
        natural_state_count = sum(
            len(item.decisions) for item in trajectories if item.source == "natural_reference"
        )
        paired_state_count = sum(
            len(item.decisions) - 1
            for item in trajectories
            if item.source in {"paired_treatment", "paired_control"}
        )
        reference_state_count = natural_state_count + paired_state_count
        value_loss_total = 0.0
        outcome_loss_totals = [0.0, 0.0]
        # This batch's legal reference states, kept (without graphs) for the
        # value head's extra fit passes and replay window.
        fit_features: list[Any] = []
        fit_task_types: list[Any] = []
        fit_rewards: list[Any] = []
        value_labels: list[tuple[int, float, int]] = []
        # Pre-update log pi_theta - log rho on decisions with a real choice.
        branch_log_ratios: dict[str, list[float]] = {}
        for trajectory in trajectories:
            snapshot = snapshots[trajectory.statistics_context]
            root = snapshot.task_anchor(
                trajectory.task.task_id,
                trajectory.task.family,
                exclude_observation_id=(
                    trajectory.sample_id if trajectory.source == "natural_reference" else None
                ),
                prior_rate=trajectory.task.prior_rate,
                prior_count=trajectory.task.prior_count,
            )
            ratios = []
            flows = []
            reference_features = []
            reference_encodings = []
            streaming = self.config.gradient_mode == "streaming"
            for index, decision in enumerate(trajectory.decisions):
                # Streaming retains scalar scores, never a trajectory's LM
                # autograd graphs. The frozen reference/encoder need no graphs
                # in either mode, including on a forced first intervention.
                with torch.no_grad() if streaming else nullcontext():
                    actor_score = self.policy.score(decision)
                with torch.no_grad():
                    combined = getattr(self.policy, "reference_score_and_encoding", None)
                    if combined is not None and not decision.reference_encoding:
                        # One adapter-free forward instead of two for the same prefix.
                        reference_score, encoding = combined(decision)
                    else:
                        reference_score = self.policy.score(decision, reference=True)
                        encoding = decision.reference_encoding or self.policy.encode_state(
                            decision.prompt_ids
                        )
                ratios.append(actor_score - reference_score)
                if len(decision.legal_token_paths) > 1:
                    branch_log_ratios.setdefault(trajectory.source, []).append(
                        float(ratios[-1].detach())
                    )
                features = torch.tensor(
                    [decision.features], dtype=torch.float32, device=self.policy.device
                )
                encoded = torch.tensor([encoding], dtype=torch.float32, device=self.policy.device)
                residual = self.heads.flow_residual(encoded, features).squeeze(0)
                measured = min(
                    self.config.beta,
                    max(
                        0.0,
                        root + snapshot.structure_correction(structure_visit(decision.state_json)),
                    ),
                )
                flows.append(residual + residual.new_tensor(measured))
                if _reference_state(trajectory, index):
                    reference_features.append(features)
                    reference_encodings.append(encoded)
            flow_values = torch.stack([*flows, flows[0].new_zeros(())])
            ratio_values = torch.stack(ratios)
            objective_ratios = (
                ratio_values.detach().requires_grad_(True) if streaming else ratio_values
            )
            objective_flows = (
                flow_values.detach().requires_grad_(True) if streaming else flow_values
            )
            loss = anchor_tb_loss(
                objective_ratios,
                objective_flows,
                reward=trajectory.reward,
                beta=self.config.beta,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite AnchorTB loss")
            if streaming:
                # This small graph differentiates the full all-subtrajectory
                # loss. Each action has its own coefficient; this is not a
                # single trajectory-balance multiplier. The root receives its
                # Eq. (8) flow coefficient, just like other nonterminal states;
                # only the observed terminal flow has a zero coefficient.
                action_coefficients, flow_coefficients = torch.autograd.grad(
                    loss / len(trajectories),
                    (objective_ratios, objective_flows),
                )
                # The residual head graphs are small and have no LM ancestors.
                flow_values.backward(flow_coefficients.detach())
                for decision, coefficient in zip(
                    trajectory.decisions,
                    action_coefficients.detach(),
                    strict=True,
                ):
                    # All parameters remain unchanged between the two passes;
                    # the policy backend keeps dropout disabled in eval mode.
                    # Backward finishes before constructing the next LM graph.
                    score = self.policy.score(decision)
                    (score * coefficient).backward()
                    del score
            else:
                (loss / len(trajectories)).backward()
            actor_loss += float(loss.detach()) / len(trajectories)
            action_count += len(trajectory.decisions)
            if reference_features:
                # Release each trajectory's diagnostic activations immediately.
                # The fraction makes this identical to MSE over all eligible
                # states in the complete batch, regardless of source or length.
                count = len(reference_features)
                fraction = count / reference_state_count
                features_batch = torch.cat(reference_features, dim=0)
                encoding_batch = torch.cat(reference_encodings, dim=0)
                targets = features_batch.new_full((count,), trajectory.reward)
                families = torch.full(
                    (count,),
                    self.task_families.index(trajectory.task.family),
                    dtype=torch.long,
                    device=self.policy.device,
                )
                value_loss = self.heads.runtime_loss(features_batch, families, targets)
                outcome_losses = self.heads.outcome_losses(encoding_batch, features_batch, targets)
                if not bool(torch.isfinite(value_loss)) or not bool(
                    torch.isfinite(outcome_losses).all()
                ):
                    raise FloatingPointError("nonfinite supervised reference loss")
                if self.config.value_loss_weight:
                    (value_loss * fraction * self.config.value_loss_weight).backward()
                if self.config.outcome_loss_weight:
                    # Mean of the two independently parameterized diagnostic
                    # MSEs: an explicit choice where Appendix B.2 is unspecified.
                    (outcome_losses.mean() * fraction * self.config.outcome_loss_weight).backward()
                value_loss_total += float(value_loss.detach()) * fraction
                fit_features.append(features_batch.detach())
                fit_task_types.append(families)
                fit_rewards.append(targets.detach())
                value_labels.append(
                    (self.task_families.index(trajectory.task.family), trajectory.reward, count)
                )
                for head_index in range(2):
                    outcome_loss_totals[head_index] += (
                        float(outcome_losses[head_index].detach()) * fraction
                    )
        parameters = (*self.policy.trainable_parameters(), *self.heads.parameters())
        for parameter in parameters:
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError("nonfinite EvoSteer gradient; update not applied")
        # Three independent optimization objectives use the same configured
        # norm limit (paper default 1). Diagnostic/value gradients must never
        # rescale an actor or residual update through a shared clipping factor.
        norm = torch.nn.utils.clip_grad_norm_(
            (*self.policy.trainable_parameters(), *self.heads.residual_head.parameters()),
            self.config.gradient_clip,
            error_if_nonfinite=True,
        )
        value_norm = torch.nn.utils.clip_grad_norm_(
            self.heads.runtime_head.parameters(),
            self.config.gradient_clip,
            error_if_nonfinite=True,
        )
        outcome_norm = torch.nn.utils.clip_grad_norm_(
            self.heads.outcome_heads.parameters(),
            self.config.gradient_clip,
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        self.policy.version += 1
        if fit_features:
            current = _ValueReplayBatch(
                trajectories[0].batch_id,
                torch.cat(fit_features).cpu(),
                torch.cat(fit_task_types).cpu(),
                torch.cat(fit_rewards).cpu(),
            )
        else:
            current = _ValueReplayBatch(
                trajectories[0].batch_id,
                torch.zeros((0, self.heads.feature_dim), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.long),
                torch.zeros((0,), dtype=torch.float32),
            )
        value_fit = self._fit_value_head(
            current,
            joint_step=bool(reference_state_count) and bool(self.config.value_loss_weight),
        )
        branch_all = [value for values in branch_log_ratios.values() for value in values]
        on_policy = branch_log_ratios.get("current", [])
        return {
            # Mean log-ratio on the actor's own samples estimates KL(pi_theta || rho).
            "policy_kl_current": sum(on_policy) / len(on_policy) if on_policy else 0.0,
            "policy_abs_log_ratio": (
                sum(abs(value) for value in branch_all) / len(branch_all) if branch_all else 0.0
            ),
            "branch_decisions": len(branch_all),
            "anchor_tb_loss": actor_loss,
            # Pre-fit MSE: this batch's reference states under the head as it
            # was before any step of this batch, i.e. before training on them.
            "value_loss": value_loss_total,
            # In-sample MSE of the per-family mean on the same states.
            "value_family_mean_loss": family_mean_loss(value_labels),
            # Post-fit MSE over the whole fit window, after the last value step.
            "value_fit_loss": value_fit["value_fit_loss"],
            "value_fit_steps": value_fit["value_fit_steps"],
            "value_fit_states": value_fit["value_fit_states"],
            "outcome_loss": sum(outcome_loss_totals) / 2,
            "outcome_1_loss": outcome_loss_totals[0],
            "outcome_2_loss": outcome_loss_totals[1],
            "gradient_norm": float(norm),
            # Joint-step pre-clip norm; the extra value passes are not included.
            "value_gradient_norm": float(value_norm),
            "outcome_gradient_norm": float(outcome_norm),
            "trajectories": len(trajectories),
            "actions": action_count,
            "natural_reference_states": natural_state_count,
            "paired_reference_states": paired_state_count,
            "supervised_reference_states": reference_state_count,
            "mean_reward": sum(item.reward for item in trajectories) / len(trajectories),
        }
