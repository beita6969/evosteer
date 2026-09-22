"""CLI wiring for formal training; runtime ownership remains in its original module."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from .training_observation import (
    observation_config,
    require_observation_sources,
    require_training_sources,
    unqualified_training_condition,
)


def main() -> None:
    from . import bayesian_improve_training as entry

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--allow-new-reasoning",
        action="store_true",
        help="Explicit public reasoning catalog/token-budget condition boundary",
    )
    parser.add_argument(
        "--allow-token-budget-notice",
        action="store_true",
        help="Explicit runtime token-cap notice at a complete checkpoint",
    )
    parser.add_argument(
        "--allow-skill-cold-start",
        action="store_true",
        help="declare zero-coverage Generate extension and optional reads at a complete checkpoint",
    )
    parser.add_argument(
        "--allow-catalog-read",
        action="store_true",
        help="Explicit catalog exposure/guidance change at a complete checkpoint",
    )
    parser.add_argument(
        "--allow-domain-subset",
        action="store_true",
        help="Remove future domain occurrences after a complete checkpoint",
    )
    parser.add_argument(
        "--allow-healthbench-judge",
        action="store_true",
        help="Declare the owner-selected external HealthBench scorer at a complete checkpoint",
    )
    parser.add_argument(
        "--allow-format-review",
        action="store_true",
        help="Declare Luna medium format-zero content review after a complete checkpoint",
    )
    parser.add_argument(
        "--iid-baselines",
        type=Path,
        help="Private paths to the two completed read-only IID Step-0 arms",
    )
    parser.add_argument("--resume", type=Path, help="Complete method snapshot from this same run")
    parser.add_argument(
        "--allow-new-action-wire",
        action="store_true",
        help="Explicit native-single-tool-call@2 to @3 boundary at a complete checkpoint",
    )
    parser.add_argument(
        "--continue-with-new-horizons",
        action="store_true",
        help="Explicitly change only static/interactive caps after a complete checkpoint",
    )
    parser.add_argument(
        "--quality-policy",
        type=Path,
        help="Frozen fixed-panel pause rules; required again when resuming a guarded run",
    )
    parser.add_argument(
        "--quality-panel",
        type=Path,
        help="Private source-disjoint panel records, exclusions and fixed sampling coordinates",
    )
    parser.add_argument(
        "--initial-quality-probe",
        type=Path,
        help="Import completed T0 ProtocolProbe unchanged; requires matching policy/panel/snapshot",
    )
    parser.add_argument(
        "--pause-at-step",
        type=int,
        help="Save and pause at this committed step without changing the full run plan",
    )
    parser.add_argument(
        "--observation-steps",
        type=int,
        choices=(5,),
        help="NEW five-step training observation; no A0 admission claim",
    )
    parser.add_argument(
        "--observation-sources",
        type=Path,
        help="Frozen canonical training allowlist and IID/development/quality exclusions",
    )
    parser.add_argument(
        "--unqualified-training-sources",
        type=Path,
        help="Explicit owner-authorized unqualified 250; no IID acceptance claim; repeat on resume",
    )
    args = parser.parse_args()
    if (
        sum(
            (
                args.allow_new_reasoning,
                args.continue_with_new_horizons,
                args.allow_new_action_wire,
                args.allow_token_budget_notice,
                args.allow_catalog_read,
                args.allow_domain_subset,
                args.allow_skill_cold_start,
                args.allow_healthbench_judge,
                args.allow_format_review,
            )
        )
        > 1
    ):
        parser.error("select only one explicit condition continuation")
    if (
        args.allow_new_action_wire
        or args.allow_token_budget_notice
        or args.allow_catalog_read
        or args.allow_domain_subset
        or args.allow_skill_cold_start
        or args.allow_healthbench_judge
        or args.allow_format_review
    ) and args.resume is None:
        parser.error("prompt condition continuation requires --resume")
    if args.initial_quality_probe is not None and (
        args.quality_policy is None or args.quality_panel is None
    ):
        parser.error("--initial-quality-probe requires --quality-policy and --quality-panel")
    from .fresh_restart import load_fresh_config, require_iid_baselines

    config = (
        entry.BayesianFormalConfig.load(args.config)
        if args.resume
        else load_fresh_config(args.config)
    )
    observation_config(
        config,
        steps=args.observation_steps,
        resume=args.resume,
        sources=args.observation_sources,
        continuation=any(
            (
                args.allow_new_reasoning,
                args.continue_with_new_horizons,
                args.allow_new_action_wire,
                args.allow_token_budget_notice,
                args.allow_catalog_read,
                args.allow_domain_subset,
                args.allow_skill_cold_start,
                args.allow_healthbench_judge,
                args.allow_format_review,
            )
        ),
    )
    if args.observation_steps is not None and args.iid_baselines is not None:
        parser.error("observation does not import or claim IID admission")
    unqualified = unqualified_training_condition(
        config,
        sources=args.unqualified_training_sources,
        root=args.run_root.resolve(),
        resume=args.resume,
        iid_baselines=args.iid_baselines,
        observation_steps=args.observation_steps,
        observation_sources=args.observation_sources,
        proactive_catalog_continuation=args.allow_catalog_read,
        domain_subset_continuation=args.allow_domain_subset,
        healthbench_judge_continuation=args.allow_healthbench_judge,
        format_review_continuation=args.allow_format_review,
        continuation=any(
            (
                args.allow_new_reasoning,
                args.continue_with_new_horizons,
                args.allow_new_action_wire,
                args.allow_token_budget_notice,
                args.allow_catalog_read,
                args.allow_domain_subset,
                args.allow_skill_cold_start,
                args.allow_healthbench_judge,
                args.allow_format_review,
            )
        ),
    )
    if args.resume is None and args.observation_steps is None and not unqualified:
        if args.iid_baselines is None:
            parser.error(
                "fresh training requires both accepted architecture-matched IID Step-0 arms"
            )
        require_iid_baselines(args.iid_baselines, config=config)
    bindings = entry.FormalTrainingBindings.load(args.bindings)
    if args.observation_steps is not None:
        records = entry.load_seven_domain_training_sources(bindings.dataset)
        require_observation_sources(
            args.observation_sources, entry.seven_domain_training_trajectories(records, steps=5)
        )
    if unqualified:
        records = entry.load_seven_domain_training_sources(bindings.dataset)
        from .training_domain_schedule import training_schedule

        selected = training_schedule(config, records, root=args.run_root, resume=args.resume)
        require_training_sources(
            args.unqualified_training_sources,
            selected,
            expected_trajectories=len(selected),
        )
    profile = entry.TrainingPerformanceConfig.load(Path(config.performance_profile))
    entry.require_formal_execution(profile)
    bindings.require_device_mapping(
        os.environ.get("CUDA_VISIBLE_DEVICES", ""), int(os.environ.get("WORLD_SIZE", "1"))
    )
    profile.configure_process()
    from .warmup_initialization import require_initialization_candidate

    require_initialization_candidate(
        bindings.preparation, sampling=config.sampling_config.to_value()
    )
    topology = entry.initialize_distributed_ttb(timeout_minutes=180)
    try:
        if topology.rank == 0:
            asyncio.run(
                entry.run_coordinator(
                    config=config,
                    bindings=bindings,
                    profile=profile,
                    root=args.run_root.resolve(),
                    resume=None if args.resume is None else args.resume.resolve(),
                    topology=topology,
                    allow_new_horizons=args.continue_with_new_horizons,
                    allow_new_reasoning=args.allow_new_reasoning,
                    allow_new_action_wire=args.allow_new_action_wire,
                    allow_token_budget_notice=args.allow_token_budget_notice,
                    allow_catalog_read=args.allow_catalog_read,
                    allow_domain_subset=args.allow_domain_subset,
                    allow_skill_cold_start=args.allow_skill_cold_start,
                    allow_healthbench_judge=args.allow_healthbench_judge,
                    allow_format_review=args.allow_format_review,
                    quality_policy=args.quality_policy,
                    quality_panel=args.quality_panel,
                    initial_quality_probe=args.initial_quality_probe,
                    pause_at_step=args.pause_at_step,
                    iid_baselines=args.iid_baselines,
                    observation_steps=args.observation_steps,
                    observation_sources=args.observation_sources,
                    unqualified_training_sources=args.unqualified_training_sources,
                )
            )
        else:
            backbone_config, checkpoint = entry._read_preparation(bindings.preparation)
            config.require_backbone(backbone_config)
            backbone = entry.build_qwen_policy_backbone(backbone_config, performance=profile)
            backbone.load_checkpoint(checkpoint.directory)
            backbone.bind_initial_trainable_state(checkpoint.trainable_state)
            entry.serve_distributed_ttb_worker(topology=topology, backbone=backbone)
    except BaseException:
        # A failed rank must reach torchrun's supervisor immediately. Collective
        # destruction here can hide its error while another rank awaits work.
        raise
    else:
        if entry.dist.is_initialized():
            entry.dist.destroy_process_group()
