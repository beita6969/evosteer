"""Owner-declared horizon changes retain old configuration and full method state."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from skillev.contracts import JsonValue
from skillev.contracts.action_wire import NATIVE_TOOL_CARRIER_WIRE, NATIVE_TOOL_HANDOFF_WIRE
from skillev.contracts.identity import validate_identifier
from skillev.contracts.skill_exposure import TWO_SKILL_CATALOG_EXPOSURE, can_continue_skill_exposure
from skillev.training.checkpoint import FilesystemTrainingCheckpointStore
from skillev.training.inflight import durable_json

from .bayesian_training_config import BayesianFormalConfig

if TYPE_CHECKING:
    from skillev.application import SKILLEVApplication


def resume_condition(
    root: Path,
    target: BayesianFormalConfig,
    *,
    allow_new_horizons: bool,
    allow_new_reasoning: bool = False,
    allow_new_action_wire: bool = False,
    allow_token_budget_notice: bool = False,
    allow_catalog_read: bool = False,
    allow_domain_subset: bool = False,
    allow_skill_cold_start: bool = False,
    allow_healthbench_judge: bool = False,
    allow_format_review: bool = False,
) -> BayesianFormalConfig | None:
    if (
        sum(
            (
                allow_new_horizons,
                allow_new_reasoning,
                allow_new_action_wire,
                allow_token_budget_notice,
                allow_catalog_read,
                allow_domain_subset,
                allow_skill_cold_start,
                allow_healthbench_judge,
                allow_format_review,
            )
        )
        > 1
    ):
        raise ValueError("select only one explicit condition continuation")
    current = root / "condition-current.json"
    value = (
        json.loads(current.read_text())["config"]
        if current.exists()
        else json.loads((root / "formal-config.json").read_text())
    )
    if value == target.to_value():
        return None
    source = BayesianFormalConfig(**value)
    horizon_change = (
        allow_new_horizons
        and replace(source, max_turns=target.max_turns, static_max_turns=target.static_max_turns)
        == target
    )
    reasoning_change = (
        allow_new_reasoning
        and replace(
            source,
            reasoning_tool_catalog=target.reasoning_tool_catalog,
            reasoning_tokens_by_domain=target.reasoning_tokens_by_domain,
            max_reasoning_tokens=target.max_reasoning_tokens,
        )
        == target
    )
    action_wire_change = (
        allow_new_action_wire
        and (source.action_wire, target.action_wire)
        == (NATIVE_TOOL_CARRIER_WIRE, NATIVE_TOOL_HANDOFF_WIRE)
        and replace(source, action_wire=target.action_wire) == target
    )
    notice_change = (
        allow_token_budget_notice
        and not source.token_budget_notice
        and target.token_budget_notice
        and replace(source, token_budget_notice=True) == target
    )
    catalog_change = (
        allow_catalog_read
        and can_continue_skill_exposure(source.skill_exposure, target.skill_exposure)
        and replace(source, skill_exposure=target.skill_exposure) == target
    )
    domain_change = (
        allow_domain_subset
        and set(target.domains) < set(source.domains)
        and replace(
            source,
            domains=target.domains,
            thinking_off_domains=tuple(
                d for d in source.thinking_off_domains if d in target.domains
            ),
            reasoning_tokens_by_domain=tuple(
                v for v in source.reasoning_tokens_by_domain if v[0] in target.domains
            ),
            action_tokens_by_domain=tuple(
                v for v in source.action_tokens_by_domain if v[0] in target.domains
            ),
        )
        == target
    )
    cold_start_change = (
        allow_skill_cold_start
        and source.cold_start is None
        and target.cold_start is not None
        and target.skill_exposure == "catalog-then-read@1"
        and replace(source, cold_start=target.cold_start, skill_exposure=target.skill_exposure)
        == target
    )
    from skillev.evaluation.external_judge_policy import (
        DIRECT_HEALTHBENCH_PROFILE,
        GATEWAY_HEALTHBENCH_PROFILE,
    )
    from skillev.evaluation.healthbench_luna_profile import LEGACY_TRAINING_JUDGE, PROFILE_ID

    judge_change = (
        allow_healthbench_judge
        and source.healthbench_judge
        in {
            LEGACY_TRAINING_JUDGE,
            DIRECT_HEALTHBENCH_PROFILE,
            GATEWAY_HEALTHBENCH_PROFILE,
        }
        and target.healthbench_judge in {GATEWAY_HEALTHBENCH_PROFILE, PROFILE_ID}
        and replace(source, healthbench_judge=target.healthbench_judge) == target
    )
    format_review_change = (
        allow_format_review
        and source.format_review_from_step is None
        and target.format_review_from_step is not None
        and replace(source, format_review_from_step=target.format_review_from_step) == target
    )
    if not (
        format_review_change
        or judge_change
        or cold_start_change
        or horizon_change
        or reasoning_change
        or action_wire_change
        or notice_change
        or catalog_change
        or domain_change
    ):
        raise ValueError("resume changes an undeclared condition field")
    return source


def save_condition_transition(
    application: SKILLEVApplication,
    *,
    root: Path,
    source_snapshot: Path,
    source: BayesianFormalConfig,
    target: BayesianFormalConfig,
    reasoning: bool = False,
    action_wire: bool = False,
    token_budget_notice: bool = False,
    catalog_read: bool = False,
    domain_subset: bool = False,
    skill_cold_start: bool = False,
    healthbench_judge: bool = False,
    format_review: bool = False,
) -> Path:
    if (
        sum(
            (
                reasoning,
                action_wire,
                token_budget_notice,
                catalog_read,
                domain_subset,
                skill_cold_start,
                healthbench_judge,
                format_review,
            )
        )
        > 1
    ):
        raise ValueError("select only one explicit condition continuation")
    step = application.training_loop.optimizer_step
    name = f"condition-step-{step:08d}-static{target.static_max_turns}-alf{target.max_turns}"
    if reasoning:
        name = f"reasoning-condition-step-{step:08d}"
    if action_wire:
        name = f"action-wire-condition-step-{step:08d}-native3"
    if token_budget_notice:
        name = f"token-budget-condition-step-{step:08d}"
    if catalog_read:
        label = (
            "skill-catalog-two-reads"
            if target.skill_exposure == TWO_SKILL_CATALOG_EXPOSURE
            else "skill-catalog"
        )
        name = f"{label}-condition-step-{step:08d}"
    if skill_cold_start:
        name = f"skill-cold-start-condition-step-{step:08d}"
    if domain_subset:
        name = f"domain-subset-condition-step-{step:08d}"
    if healthbench_judge:
        name = f"healthbench-judge-condition-step-{step:08d}"
    if format_review:
        if target.format_review_from_step != step + 1:
            raise ValueError("format review must start immediately after its source checkpoint")
        name = f"format-review-condition-step-{step:08d}"
    directory = root / "checkpoints" / name
    declaration: dict[str, JsonValue] = {
        "format": "action-wire-condition-transition@1"
        if action_wire
        else "reasoning-condition-transition@1"
        if reasoning
        else "horizon-condition-transition@1",
        "sampling_condition": application.snapshot_identity.sampling_schedule_algorithm,
        "source_checkpoint": str(source_snapshot),
        "source_config": source.to_value(),
        "config": target.to_value(),
        "saved_optimizer_step": step,
        "effective_from_optimizer_step": step + 1,
        "historical_evidence": "preserved-with-original-labels-weights-and-horizon-features",
        "checkpoint": str(directory),
    }
    if token_budget_notice:
        declaration["format"] = "token-budget-notice-condition-transition@1"
    if catalog_read:
        declaration["format"] = "skill-exposure-condition-transition@1"
        declaration["historical_evidence"] = (
            "preserved-with-original-skill-exposure-and-invocation-credit"
        )
    if action_wire:
        declaration["historical_evidence"] = (
            "preserved-with-original-labels-weights-and-action-wire-identities"
        )
    if skill_cold_start:
        declaration["format"] = "skill-cold-start-condition-transition@1"
        declaration["method_extension"] = (
            target.cold_start.to_value() if target.cold_start else None
        )
        declaration["historical_evidence"] = (
            "preserved-no-retroactive-invocations-or-new-skill-credit"
        )
    if healthbench_judge:
        declaration["format"] = "healthbench-judge-condition-transition@1"
        declaration["historical_evidence"] = (
            "preserved-original-rewards-and-posterior-no-retrospective-regrading"
        )
    if domain_subset:
        declaration["format"] = "domain-subset-condition-transition@1"
        declaration["historical_evidence"] = "consumed-prefix-and-complete-learning-state-preserved"
        declaration["source_batch_size"] = source.batch_size
        declaration["batch_size"] = target.batch_size
        declaration["removed_domains"] = [d for d in source.domains if d not in target.domains]
    if format_review:
        declaration["format"] = "format-review-condition-transition@1"
        declaration["historical_evidence"] = (
            "original-native-labels-preserved-no-retrospective-review"
        )
    path = root / f"{name}.json"
    if path.exists():
        if json.loads(path.read_text()) != declaration:
            raise ValueError("condition boundary already records a different transition")
    else:
        durable_json(path, declaration)
    if directory.exists():
        metadata = FilesystemTrainingCheckpointStore(root=root / "checkpoints").load_metadata(
            directory
        )
        if metadata.optimizer_step != step or metadata.identity != application.snapshot_identity:
            raise ValueError("saved condition boundary differs from the restored application")
    else:
        application.evolution_loop.save_condition_boundary(name)
    # Publish only after model, optimizer and all method state are durable.
    durable_json(root / "condition-current.json", declaration)
    return directory


def sampling_condition(root: Path, fallback: str) -> str:
    """Rendering changes do not change the fixed task/seed curriculum identity."""
    for name in ("condition-current.json", "branch-source.json"):
        path = root / name
        if path.exists():
            value = json.loads(path.read_text()).get("sampling_condition")
            if isinstance(value, str) and value:
                return value
    return fallback


def condition_configs(root: Path) -> dict[int, BayesianFormalConfig]:
    """Published conditions only; retain the original configuration at each boundary."""
    source = BayesianFormalConfig(**json.loads((root / "formal-config.json").read_text()))
    starts = {1: source}
    current = root / "condition-current.json"
    if current.exists():
        published = json.loads(current.read_text())
        through = published["effective_from_optimizer_step"]
        declarations = [
            json.loads(path.read_text()) for path in root.glob("*condition-step-*.json")
        ]
        for row in sorted(declarations, key=lambda value: value["effective_from_optimizer_step"]):
            step = row["effective_from_optimizer_step"]
            if step > through:
                continue  # A boundary not yet published must not relabel evidence.
            if row["source_config"] != source.to_value() or step != row["saved_optimizer_step"] + 1:
                raise ValueError("condition history differs from the saved continuation chain")
            source = BayesianFormalConfig(**row["config"])
            starts[step] = source
        if published["config"] != source.to_value():
            raise ValueError("published condition is missing its history")
    return starts


def observer_condition_starts(root: Path, target: BayesianFormalConfig) -> dict[int, str]:
    starts = condition_configs(root)
    if starts[max(starts)] != target:
        raise ValueError("observer condition has not been published")
    return {step: config.condition for step, config in starts.items()}


def observer_batch_size_starts(root: Path, target: BayesianFormalConfig) -> dict[int, int]:
    starts = condition_configs(root)
    if starts[max(starts)] != target:
        raise ValueError("observer batch-size condition has not been published")
    return {step: config.batch_size for step, config in starts.items()}


def _bind_run_directory(
    root: Path,
    config: BayesianFormalConfig,
    resume: Path | None,
    *,
    allow_new_horizons: bool = False,
    allow_new_reasoning: bool = False,
    allow_new_action_wire: bool = False,
    allow_token_budget_notice: bool = False,
    allow_catalog_read: bool = False,
    allow_domain_subset: bool = False,
    allow_skill_cold_start: bool = False,
    allow_healthbench_judge: bool = False,
    allow_format_review: bool = False,
) -> BayesianFormalConfig | None:
    if (
        sum(
            (
                allow_new_horizons,
                allow_new_reasoning,
                allow_new_action_wire,
                allow_token_budget_notice,
                allow_catalog_read,
                allow_domain_subset,
                allow_skill_cold_start,
                allow_healthbench_judge,
                allow_format_review,
            )
        )
        > 1
    ):
        raise ValueError("select only one explicit condition continuation")
    if (
        allow_new_action_wire
        or allow_token_budget_notice
        or allow_catalog_read
        or allow_domain_subset
        or allow_skill_cold_start
        or allow_healthbench_judge
        or allow_format_review
    ) and resume is None:
        raise ValueError("prompt condition continuation requires a complete source checkpoint")
    validate_identifier(root.name)
    if resume is None:
        root.mkdir(parents=True, mode=0o700, exist_ok=False)
        durable_json(root / "formal-config.json", config.to_value())
        return None
    return resume_condition(
        root,
        config,
        allow_new_horizons=allow_new_horizons,
        allow_new_reasoning=allow_new_reasoning,
        allow_new_action_wire=allow_new_action_wire,
        allow_token_budget_notice=allow_token_budget_notice,
        allow_catalog_read=allow_catalog_read,
        allow_domain_subset=allow_domain_subset,
        allow_skill_cold_start=allow_skill_cold_start,
        allow_healthbench_judge=allow_healthbench_judge,
        allow_format_review=allow_format_review,
    )
