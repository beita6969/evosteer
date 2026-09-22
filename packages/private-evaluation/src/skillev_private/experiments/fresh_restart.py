"""Explicit new-run controls; legacy checkpoints keep their historical defaults."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, cast

import yaml

from skillev.contracts import JsonValue
from skillev.evaluation.healthbench_luna_profile import healthbench_condition
from skillev.policy import QwenMultimodalBackboneConfig
from skillev.rollout import RolloutTask
from skillev.runtime import SkillLibraryState
from skillev.training.run_condition import EffectiveRunCondition
from skillev_private.benchmarks.protocol_v13_seven_training import SEVEN_TRAINING_DOMAINS

from .bayesian_training_config import BayesianFormalConfig

if TYPE_CHECKING:
    from skillev.application import SKILLEVApplication


def require_clean_initial_application(
    application: SKILLEVApplication,
    *,
    preparation: Path,
    checkpoint_directory: Path,
    root: Path,
    initial_skill_profile: str = "public-advisory@2",
) -> None:
    """Use independently saved initial trainables, never the candidate as its own reference."""
    import torch

    from skillev.experiments._evolution_preflight_seed import planned_seed_documents
    from skillev.policy.checkpoint import read_policy_checkpoint_state
    from skillev.training.fresh_state import (
        FreshNamespaces,
        inspect_fresh_state,
        require_fresh_state,
        save_fresh_state_report,
    )

    from .warmup_initialization import initialization_condition, require_warmup_initial_application

    if initialization_condition(preparation):
        require_warmup_initial_application(
            application,
            preparation=preparation,
            checkpoint_directory=checkpoint_directory,
            root=root,
            initial_library=SkillLibraryState.from_seed_documents(
                planned_seed_documents(initial_skill_profile)
            ),
        )
        return

    parameters = torch.load(
        preparation.parent / "initial_named_parameters.pt", map_location="cpu", weights_only=True
    )
    if not isinstance(parameters, dict) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in parameters.items()
    ):
        raise ValueError("fresh preparation requires its original named trainable tensors")
    report = inspect_fresh_state(
        application,
        initial_library=SkillLibraryState.from_seed_documents(
            planned_seed_documents(initial_skill_profile)
        ),
        namespaces=FreshNamespaces(
            request_journals=(root / "requests.sqlite3",),
            evidence_directories=(root / "inflight", root / "evidence"),
        ),
        preparation_state=read_policy_checkpoint_state(checkpoint_directory),
        initial_parameters=parameters,
    )
    save_fresh_state_report(root / "fresh-application-start.json", report)
    require_fresh_state(report)


def resolved_input_profiles(tasks: tuple[RolloutTask, ...]) -> dict[str, str]:
    from skillev.task_semantic_guidance import TRAINING_PUBLIC_INPUT

    profiles: dict[str, str] = {}
    for task in tasks:
        context = task.public_context
        if not isinstance(context, dict):
            raise ValueError("fresh task requires declared public source semantics")
        domain, profile = (
            context.get("benchmark_id"),
            context.get("input_profile", TRAINING_PUBLIC_INPUT),
        )
        if not isinstance(domain, str) or not isinstance(profile, str):
            raise ValueError("fresh task requires a domain and public input profile")
        if domain in profiles and profiles[domain] != profile:
            raise ValueError("one domain cannot silently mix public input profiles")
        profiles[domain] = profile
    return profiles


def require_iid_baselines(
    path: Path,
    *,
    config: BayesianFormalConfig,
) -> dict[str, object]:
    """Require actual accepted A0 results for both axes, not a native-score note.

    This is the user-requested zero-update architecture prerequisite. It is not
    an attestation system, nor permission to copy IID artifacts into training.
    """
    from skillev.rollout.readonly_collection import evaluation_isolation

    paths = json.loads(path.read_text())
    if set(paths) != {"skills-off", "initial-library"}:
        raise ValueError("fresh training needs skills-off and initial-library IID Step-0 results")
    references = {}
    for arm, directory in paths.items():
        root = Path(directory)
        summary = json.loads((root / "summary.json").read_text())
        expanded = json.loads((root / "expanded-controls-private.json").read_text())
        if (
            summary.get("record_kind") != "iid-evaluation"
            or summary.get("policy_step") != 0
            or summary.get("arm") != arm
            or summary.get("baseline_accepted") is not True
            or summary.get("execution_validation") != "live-components-unchanged"
            or set(summary.get("domains", {})) != set(config.domains)
        ):
            raise ValueError("IID Step-0 is incomplete, unaccepted or from another condition")
        if any(summary.get(k) != v for k, v in evaluation_isolation().items()):
            raise ValueError("IID baseline was not collected as read-only evaluation")
        for row in summary["domains"].values():
            if (
                not row["planned"]
                or row["completed"] != row["planned"]
                or row["accepted"] is not True
            ):
                raise ValueError("IID baseline must account for every native panel ID")
        if expanded["controls"]["sampling"] != json.loads(
            json.dumps(config.sampling_config.to_value())
        ):
            raise ValueError(
                "formal sampling/phase/tool semantics differ from the accepted IID architecture"
            )
        if expanded["controls"]["formal"] != config.expanded_value():
            raise ValueError("formal condition changed after IID architecture acceptance")
        if summary["policy_snapshot_id"] != expanded["policy_snapshot"]["snapshot_id"]:
            raise ValueError("IID summary does not describe its expanded policy controls")
        references[arm] = expanded
    off, library = references["skills-off"], references["initial-library"]
    if any(
        off[k] != library[k]
        for k in ("architecture_id", "panel", "controls", "policy_snapshot", "acceptance_rules")
    ):
        raise ValueError("Step-0 arms must share one population, policy and execution architecture")
    return cast(dict[str, object], library)


def require_run_iid_baselines(
    *, requested: Path | None, root: Path, config: BayesianFormalConfig, resuming: bool
) -> dict[str, object]:
    """A resume retains the original A0 controls instead of bypassing them."""
    if resuming:
        from .warmup_initialization import require_initial_state_report

        saved = root / "iid-baselines-private.json"
        if requested is not None and json.loads(requested.read_text()) != json.loads(
            saved.read_text()
        ):
            raise ValueError("resume cannot replace this run's original IID baselines")
        require_initial_state_report(
            json.loads((root / "fresh-application-start.json").read_text())
        )
        requested = saved
    if requested is None:
        raise ValueError("fresh training requires both architecture-matched IID Step-0 arms")
    return require_iid_baselines(requested, config=config)


def load_fresh_config(path: Path) -> BayesianFormalConfig:
    raw = yaml.safe_load(path.read_text())
    # These additions have serialized legacy defaults. Requiring them in an
    # already frozen @5/@7 file would break an otherwise unchanged condition.
    names = {item.name for item in fields(BayesianFormalConfig)} - {
        "initial_skill_profile",
        "action_tokens_by_domain",
        "learning_protocol",
        "domains",
        "format_review_from_step",
    }
    if not isinstance(raw, dict) or names - raw.keys():
        raise ValueError("fresh restart must explicitly resolve every configuration control")
    if raw.get("task_semantic_guidance") == "public-task-semantics@8" and (
        "initial_skill_profile" not in raw
    ):
        raise ValueError("the skill-value candidate must explicitly choose its initial library")
    config = BayesianFormalConfig.load(path)
    require_fresh_interface(config)
    return config


def require_restart_inputs(
    reference: dict[str, object],
    *,
    backbone: QwenMultimodalBackboneConfig,
    library: SkillLibraryState,
    scorer_contracts: dict[str, JsonValue],
    training_sources: frozenset[tuple[str, str]],
) -> None:
    """Catch accidental source/library/scorer substitution after the accepted A0."""
    from skillev_private.evaluation.iid_episode_sources import canonical_source_key

    controls = cast(dict[str, object], reference["controls"])
    model = cast(dict[str, object], controls["model"])
    if model["backbone"] != backbone.to_value() or controls["scorers"] != scorer_contracts:
        raise ValueError("fresh model or native scoring differs from the accepted IID architecture")
    if reference["library_snapshot"] != library.to_value():
        raise ValueError("fresh library differs from the actual initial-library IID arm")
    panel = cast(dict[str, object], reference["panel"])
    exclusions = cast(dict[str, list[list[str]]], panel["excluded_sources"])
    aliases = cast(dict[str, dict[str, str]], panel.get("source_aliases", {}))
    actual = {canonical_source_key(row, aliases) for row in training_sources}
    excluded = {canonical_source_key((row[0], row[1]), aliases) for row in exclusions["training"]}
    if not actual <= excluded:
        raise ValueError("actual training sources were not excluded when the IID panel was frozen")


def require_fresh_interface(config: BayesianFormalConfig) -> None:
    from skillev.contracts.skill_exposure import AUTONOMOUS_CATALOG_EXPOSURES

    if (
        config.format != "skillev-bayesian-formal-training@6"
        or not config.phase_context
        or not config.reasoning_tool_catalog
        or not config.token_budget_notice
        or not config.public_action_semantics
        or config.action_wire != "native-single-tool-call@3"
        or config.skill_exposure not in AUTONOMOUS_CATALOG_EXPOSURES
        or config.task_semantic_guidance
        not in {"public-task-semantics@5", "public-task-semantics@7", "public-task-semantics@8"}
    ):
        raise ValueError(
            "fresh restart requires the declared autonomous native catalog architecture"
        )
    if not set(config.domains) <= set(dict(config.reasoning_tokens_by_domain)):
        raise ValueError("fresh restart must explicitly declare every active reasoning budget")


def resolve_effective_run_condition(
    *,
    config: BayesianFormalConfig,
    backbone: QwenMultimodalBackboneConfig,
    initial_library: SkillLibraryState,
    data_condition: dict[str, JsonValue],
    input_profiles: dict[str, str],
    scorer_contracts: dict[str, JsonValue],
    execution: dict[str, JsonValue],
    condition_id: str,
) -> EffectiveRunCondition:
    require_fresh_interface(config)
    config.require_backbone(backbone)
    if not data_condition or not isinstance(data_condition.get("ordered_selected_sources"), list):
        raise ValueError("fresh run requires its actual ordered source and split declaration")
    known_domains = {d.value for d in SEVEN_TRAINING_DOMAINS}
    if not set(config.domains) <= set(input_profiles) <= known_domains or not all(
        input_profiles.values()
    ):
        raise ValueError("every domain requires an explicit public input profile")
    if not set(config.domains) <= set(scorer_contracts) <= known_domains or not all(
        scorer_contracts.values()
    ):
        raise ValueError("every domain requires its actual terminal scorer contract")
    if scorer_contracts["healthbench"] != healthbench_condition(config.healthbench_judge):
        raise ValueError("HealthBench runtime judge differs from the frozen condition")
    declared = config.expanded_value()
    # Placement and throughput targets are not learning or evaluation controls.
    for name in ("performance_profile", "planning_hours", "target_steps_per_hour"):
        declared.pop(name)
    return EffectiveRunCondition.create(
        condition_id=condition_id,
        scientific={
            "formal": declared,
            "resolved_domains": config.schedule_summary(),
            "input_profiles": {d: input_profiles[d] for d in config.domains},
            "terminal_scorers": {d: scorer_contracts[d] for d in config.domains},
            "data_condition": data_condition,
            "initial_library": initial_library.to_value(),
            "initial_model": {
                "base_model": config.model,
                "base_dtype": config.base_dtype,
                "revision": backbone.revision,
                "tokenizer_id": backbone.tokenizer_id,
                "lora_target_modules": list(backbone.lora_target_modules),
                "z_initialization": backbone.z_initialization.to_value(),
                "seed": config.seed,
            },
        },
        execution=execution,
    )
