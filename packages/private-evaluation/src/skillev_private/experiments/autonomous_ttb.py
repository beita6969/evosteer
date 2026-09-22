"""Declared pure-TTB input condition, not another trainer or reward function.

Historical teaching and supervised initializations remain readable elsewhere.
This branch uses the existing complete F/B/Z training and continuation path.
Source roles describe public task needs; they never filter sampled trajectories
or alter a loss, reward, posterior weight, or the phase transition criterion.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from skillev.contracts.skill_exposure import AUTONOMOUS_CATALOG_EXPOSURES
from skillev_private.benchmarks.protocol_v13_seven_training import SEVEN_TRAINING_DOMAINS
from skillev_private.benchmarks.protocol_v13_training import Protocol13TrainingRecord
from skillev_private.evaluation.iid_episode_sources import canonical_source_key

from .skill_practice import CONTEXT_KEY

if TYPE_CHECKING:
    from .bayesian_training_config import BayesianFormalConfig

PROTOCOL = "autonomous-ttb@1"
SOURCE_FORMAT = "public-task-needs@1"
SOURCE_ROLES = ("direct-control", "procedure-applicable", "exploration")


def require_autonomous_config(config: BayesianFormalConfig) -> None:
    if config.learning_protocol != PROTOCOL:
        raise ValueError("unknown declared learning protocol")
    if config.cold_start is not None:
        raise ValueError("pure TTB cannot enable the zero-coverage phase-boundary extension")
    if (
        not config.format.endswith("@6")
        or not config.phase_context
        or not config.reasoning_tool_catalog
        or config.skill_exposure not in AUTONOMOUS_CATALOG_EXPOSURES
        or config.action_wire != "native-single-tool-call@3"
    ):
        raise ValueError("autonomous TTB requires the declared visible native catalog interface")
    if config.steps - config.closure_steps < 2 * config.window:
        raise ValueError("declare a bounded real TTB budget covering both diagnostic windows")


def require_autonomous_initialization(config: BayesianFormalConfig, preparation: Path) -> None:
    """Reject the actual specialized-SFT preparation, not just its display label.

    The normal preparation reader still validates the initial partition binding;
    normal resume restores the complete application checkpoint, not F alone.
    No model/optimizer is loaded or rewritten here.
    """
    if config.learning_protocol != PROTOCOL:
        return
    raw = json.loads(preparation.read_text(encoding="utf-8"))
    if (
        raw.get("format") != "skillev-private-protocol13-training-debug-preparation@1"
        or "initialization" in raw
    ):
        raise ValueError("autonomous TTB starts from the saved pre-skill-SFT initialization")


def _source(row: Any, aliases: Any) -> tuple[str, str]:
    if (
        not isinstance(row, dict)
        or not isinstance(row.get("benchmark"), str)
        or not isinstance(row.get("source_id"), str)
        or not row["source_id"].strip()
    ):
        raise ValueError("source rows require benchmark and canonical source_id")
    key = row["benchmark"], row["source_id"]
    if canonical_source_key(key, aliases) != key:
        raise ValueError("freeze canonical source coordinates, not population aliases")
    return key


def autonomous_training_sources(
    config: BayesianFormalConfig,
    records: tuple[Protocol13TrainingRecord, ...],
    data_condition: dict[str, Any] | None,
) -> tuple[Protocol13TrainingRecord, ...]:
    """Apply a pre-outcome source schedule; preserve every later sampled outcome.

    The inline declaration is persisted by the existing effective run condition,
    including the full expanded seven-domain schedule. Resume already compares
    that condition. Roles are private reporting metadata, never model prompts.
    """
    if config.learning_protocol != PROTOCOL:
        return records
    require_autonomous_config(config)
    declaration = (data_condition or {}).get("autonomous_ttb_sources")
    if not isinstance(declaration, dict) or declaration.get("format") != SOURCE_FORMAT:
        raise ValueError("declare the public-needs source schedule before autonomous sampling")
    aliases = declaration.get("source_aliases", {})
    if not isinstance(aliases, dict) or any(
        not isinstance(mapping, dict)
        or any(
            not isinstance(k, str) or not isinstance(v, str) or mapping.get(v, v) != v
            for k, v in mapping.items()
        )
        for mapping in aliases.values()
    ):
        raise ValueError("source aliases must resolve directly to canonical identities")
    exclusions = declaration.get("excluded_sources")
    if not isinstance(exclusions, dict) or not {"iid", "development", "quality"} <= set(exclusions):
        raise ValueError("keep IID, development and quality source exclusions explicit")
    excluded = set()
    for rows in exclusions.values():
        if not isinstance(rows, list):
            raise ValueError("excluded source groups must be arrays")
        for row in rows:
            if (
                not isinstance(row, list | tuple)
                or len(row) != 2
                or any(not isinstance(part, str) or not part.strip() for part in row)
            ):
                raise ValueError("excluded coordinates must be benchmark/source pairs")
            excluded.add(canonical_source_key((row[0], row[1]), aliases))
    rows = declaration.get("ordered_sources")
    if not isinstance(rows, list) or not rows:
        raise ValueError("declare the complete nonempty source order before collecting outcomes")
    available: dict[tuple[str, str], Protocol13TrainingRecord] = {}
    for record in records:
        key = canonical_source_key(
            (record.episode.benchmark.value, record.episode.source_id), aliases
        )
        if key in available:
            # Historical acquisition files can contain pre-expanded repeats.
            # Collapse identical source occurrences before freezing this NEW
            # schedule, but never choose between conflicting source variants.
            previous = available[key]
            if (
                replace(record.input, task_id=previous.input.task_id) != previous.input
                or record.output != previous.output
            ):
                raise ValueError("one canonical source has conflicting public/scoring inputs")
            continue
        available[key] = record
    selected = []
    seen = set()
    roles: Counter[str] = Counter()
    for row in rows:
        key = _source(row, aliases)
        if key in seen or key in excluded or key not in available:
            raise ValueError("declared source is duplicate, held out, or unavailable")
        if (
            row.get("role") not in SOURCE_ROLES
            or not isinstance(row.get("public_basis"), str)
            or not row["public_basis"].strip()
        ):
            raise ValueError("each source needs a pre-outcome public applicability basis")
        if row["role"] == "procedure-applicable" and (
            not isinstance(row.get("method_family"), str) or not row["method_family"].strip()
        ):
            raise ValueError("applicable tasks must name the reusable method, not a success label")
        record = available[key]
        context = record.input.public_context
        if isinstance(context, dict) and CONTEXT_KEY in context:
            raise ValueError("teaching/practice instructions cannot enter autonomous TTB")
        selected.append(record)
        seen.add(key)
        roles[row["role"]] += 1
    if not set(config.scheduled_domains) <= {r.episode.benchmark for r in selected}:
        raise ValueError("the fixed source plan must contain every active training domain")
    if not roles["direct-control"] or not roles["procedure-applicable"]:
        raise ValueError("retain both reasonable direct tasks and publicly applicable tasks")
    return tuple(selected)


def source_coverage_report(data_condition: dict[str, Any]) -> dict[str, Any]:
    """Read-only public-need counts; applicability is not proven skill benefit."""
    rows = data_condition["autonomous_ttb_sources"]["ordered_sources"]
    return {
        "selection_basis": SOURCE_FORMAT,
        "source_count": len(rows),
        "roles": dict(Counter(row["role"] for row in rows)),
        "domains": {
            domain.value: dict(
                Counter(row["role"] for row in rows if row["benchmark"] == domain.value)
            )
            for domain in SEVEN_TRAINING_DOMAINS
        },
        "skill_benefit_established": False,
        "outcome_filtering": False,
        "extra_training_objective": False,
    }
