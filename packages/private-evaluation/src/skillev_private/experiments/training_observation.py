"""Explicit five-step observation using the unchanged three-role formal runtime.

This authorizes neither IID acceptance nor an arbitrary short-run/resume bypass.
Source declarations contain coordinates only; collection and native labels stay
with the normal training implementation.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from skillev.contracts import JsonValue
from skillev_private.benchmarks.protocol_v13_training import Protocol13TrainingRecord
from skillev_private.evaluation.iid_episode_sources import canonical_source_key

from .bayesian_training_config import BayesianFormalConfig
from .fresh_restart import require_fresh_interface


def observation_config(
    config: BayesianFormalConfig,
    *,
    steps: int | None,
    resume: Path | None,
    sources: Path | None,
    continuation: bool = False,
) -> BayesianFormalConfig:
    if steps is None:
        if sources is not None:
            raise ValueError("observation sources require the explicit five-step mode")
        return config
    if type(steps) is not int or steps != 5 or resume is not None or continuation:
        raise ValueError("five-step observation is only an explicit NEW run, not a continuation")
    require_fresh_interface(config)
    if sources is None or not sources.is_file():
        raise ValueError("five-step observation requires its frozen source isolation declaration")
    if config.batch_size != 28 or config.maximum_cycles != 2:
        raise ValueError("observation preserves B28 and the existing two-cycle bound")
    return replace(config, steps=5, closure_steps=1, checkpoint_every=1)


def observation_condition(steps: int | None) -> dict[str, JsonValue]:
    if steps is None:
        return {}
    if type(steps) is not int or steps != 5:
        raise ValueError("only the authorized five-step observation is supported")
    return {
        "training_observation": {
            "format": "five-step-training-observation@1",
            "a0_admission": "not-claimed",
            "accepted_250": False,
            "optimizer_steps": 5,
            "batch_size": 28,
            "checkpoint_every": 1,
            "gradient_owners": 2,
            "coordinator_participates": True,
            "phase_evidence": "read-actual-events-no-W50-natural-trigger-claim",
        }
    }


def _coordinates(rows: object, aliases: Any) -> set[tuple[str, str]]:
    if not isinstance(rows, list) or any(
        not isinstance(row, list)
        or len(row) != 2
        or any(not isinstance(part, str) or not part.strip() for part in row)
        for row in rows
    ):
        raise ValueError("source coordinates must be benchmark/source pairs")
    return {canonical_source_key((row[0], row[1]), aliases) for row in rows}


def require_training_sources(
    path: Path,
    selected: tuple[Protocol13TrainingRecord, ...],
    *,
    expected_trajectories: int,
) -> dict[str, JsonValue]:
    """Check the actual fixed training schedule, not renamed populations/tasks."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("observation source declaration must be an object")
    exclusions = value.get("excluded_sources")
    if not isinstance(exclusions, dict) or any(
        name not in exclusions for name in ("iid", "development", "quality")
    ):
        raise ValueError("declare IID, development and quality exclusions explicitly")
    aliases = value.get("source_aliases", {})

    allowed = _coordinates(value.get("training"), aliases)
    excluded = set().union(*(_coordinates(rows, aliases) for rows in exclusions.values()))
    actual = {
        canonical_source_key((r.episode.benchmark.value, r.episode.source_id), aliases)
        for r in selected
    }
    if not actual or not actual <= allowed or actual & excluded or allowed & excluded:
        raise ValueError(
            "observation sources overlap excluded material or lack training provenance"
        )
    if len(selected) != expected_trajectories:
        raise ValueError("source declaration requires the complete declared B28 schedule")
    return cast(dict[str, JsonValue], value)


def require_observation_sources(
    path: Path, selected: tuple[Protocol13TrainingRecord, ...]
) -> dict[str, JsonValue]:
    return require_training_sources(path, selected, expected_trajectories=140)


def unqualified_training_condition(
    config: BayesianFormalConfig,
    *,
    sources: Path | None,
    root: Path,
    resume: Path | None,
    iid_baselines: Path | None = None,
    observation_steps: int | None = None,
    observation_sources: Path | None = None,
    continuation: bool = False,
    proactive_catalog_continuation: bool = False,
    domain_subset_continuation: bool = False,
    healthbench_judge_continuation: bool = False,
    format_review_continuation: bool = False,
) -> dict[str, JsonValue]:
    """Explicit owner-authorized nonqualified 250, never a successful IID result.

    A same-run resume must repeat this declaration. Ordinary/short runs cannot
    acquire this exception through resume; existing config/snapshot checks stay.
    """
    saved = root / "unqualified-training-condition.json"
    if sources is None:
        if saved.exists():
            raise ValueError("resume must explicitly retain this run's unqualified condition")
        return {}
    if (
        iid_baselines is not None
        or observation_steps is not None
        or observation_sources is not None
    ):
        raise ValueError("unqualified training cannot claim IID or five-step admission")
    resume_source = None
    if continuation:
        from skillev.contracts.skill_exposure import (
            CATALOG_EXPOSURE_VERSION,
            PROACTIVE_CATALOG_EXPOSURE,
        )

        from .bayesian_condition_transition import resume_condition

        if (
            sum(
                (
                    proactive_catalog_continuation,
                    domain_subset_continuation,
                    healthbench_judge_continuation,
                    format_review_continuation,
                )
            )
            > 1
        ):
            raise ValueError("select one explicit condition continuation")
        if format_review_continuation:
            if resume is None:
                raise ValueError("format review requires the complete source checkpoint")
            resume_source = resume_condition(
                root, config, allow_new_horizons=False, allow_format_review=True
            )
        elif healthbench_judge_continuation:
            if resume is None:
                raise ValueError("Judge change requires the complete source checkpoint")
            resume_source = resume_condition(
                root, config, allow_new_horizons=False, allow_healthbench_judge=True
            )
        elif domain_subset_continuation:
            if resume is None:
                raise ValueError("domain removal requires the complete source checkpoint")
            resume_condition(root, config, allow_new_horizons=False, allow_domain_subset=True)
        elif (
            not proactive_catalog_continuation
            or resume is None
            or config.skill_exposure != PROACTIVE_CATALOG_EXPOSURE
        ):
            raise ValueError("unqualified authorization does not authorize a method transition")
        # The caller must explicitly select the prompt-only continuation. Reuse
        # the existing comparison: no budget, source, scorer or learning change.
        if proactive_catalog_continuation:
            source = resume_condition(
                root, config, allow_new_horizons=False, allow_catalog_read=True
            )
            resume_source = source
            if source is not None and source.skill_exposure != CATALOG_EXPOSURE_VERSION:
                raise ValueError("proactive continuation requires the original autonomous catalog")
    require_fresh_interface(config)
    if (
        config.steps,
        config.closure_steps,
        config.checkpoint_every,
        config.maximum_cycles,
    ) != (250, 1, 10, 2):
        raise ValueError("unqualified training retains the full 250 / cadence10 plan")
    value = json.loads(sources.read_text())
    if (
        not isinstance(value, dict)
        or value.get("format") != "owner-authorized-unqualified-training@1"
        or value.get("authorization") != "owner-explicit"
    ):
        raise ValueError("explicit owner-authorized unqualified source declaration required")
    aliases = value.get("source_aliases", {})
    exclusions = value.get("excluded_sources")
    if not isinstance(exclusions, dict) or any(
        k not in exclusions for k in ("iid", "development", "quality")
    ):
        raise ValueError("declare IID, development and quality exclusions explicitly")
    canonical = sorted(_coordinates(value.get("training"), aliases))
    excluded = {k: sorted(_coordinates(rows, aliases)) for k, rows in exclusions.items()}
    condition: dict[str, JsonValue] = {
        "unqualified_training": {
            "format": "owner-authorized-unqualified-training@1",
            "authorization": "owner-explicit",
            "qualification": "unqualified",
            "cold_start_failed": True,
            "a0_admission": "not-claimed",
            "accepted_iid_baseline": False,
            "optimizer_steps": 250,
            "phase_search_steps": 249,
            "closure_steps": 1,
            "batch_size": config.batch_size,
            "checkpoint_every": 10,
            "source_declaration": value,
            "canonical_training_allowlist": [list(row) for row in canonical],
            "canonical_excluded_sources": {
                k: [list(row) for row in rows] for k, rows in excluded.items()
            },
        }
    }
    if resume is not None:
        original_condition = json.loads(saved.read_text()) if saved.is_file() else None
        comparable = json.loads(json.dumps(condition))
        if (
            original_condition is not None
            and config.batch_size != original_condition["unqualified_training"]["batch_size"]
        ):
            from .bayesian_condition_transition import condition_configs

            history = condition_configs(root)
            if not domain_subset_continuation and history[max(history)] != (
                resume_source or config
            ):
                raise ValueError("batch-size change lacks a published domain continuation")
            comparable["unqualified_training"]["batch_size"] = original_condition[
                "unqualified_training"
            ]["batch_size"]
        if original_condition != comparable:
            raise ValueError("resume cannot add/remove/change the unqualified source condition")
        original = root / "unqualified-training-sources-private.json"
        if not original.is_file() or json.loads(original.read_text()) != value:
            raise ValueError("resume lacks the unchanged original unqualified source declaration")
    return condition
