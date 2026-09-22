"""Frozen seven-IID execution controls and policy/library-only interventions.

The bundle is private evaluation data, not a training configuration or batch.
It holds native source targets separately from public tasks. No model is loaded
and no population is chosen from scores by this module.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from skillev.contracts import normalize_json
from skillev.evaluation.healthbench_luna_profile import healthbench_condition
from skillev.rollout import PolicySnapshot, RolloutTask
from skillev.rollout.readonly_collection import evaluation_isolation
from skillev.runtime import SkillLibraryState
from skillev.training.inflight import durable_json
from skillev.training.rollout_workflow import RolloutWorkflowBinding
from skillev_private.benchmarks.evaluation_episode import (
    EvaluationEpisodeRecord,
    EvaluationSource,
    EvaluationTarget,
)
from skillev_private.benchmarks.protocol_v13_seven_training import SEVEN_TRAINING_DOMAINS
from skillev_private.experiments.bayesian_training_config import BayesianFormalConfig
from skillev_private.experiments.formal_episode_config import formal_actor_transport
from skillev_private.experiments.fresh_restart import require_fresh_interface

from .iid_episode_sources import (
    IIDSourceIdentity,
    canonical_source_key,
    evaluation_records,
    require_source_isolation,
)
from .integrity_sources import SourcePanel

FORMAT = "architecture-matched-iid@1"
ARMS = ("skills-off", "initial-library")


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def record_value(record: EvaluationEpisodeRecord) -> dict[str, Any]:
    return {
        "purpose": "evaluation-only",
        "source": asdict(record.episode),
        "public_task": record.input.to_value(),
        "native_target": record.output.target,
    }


def decode_record(value: dict[str, Any]) -> EvaluationEpisodeRecord:
    from skillev.evaluation.current_iid.protocol13.catalog import Protocol13Benchmark

    if value["purpose"] != "evaluation-only":
        raise ValueError("IID execution cannot load a training record")
    source = value["source"]
    return EvaluationEpisodeRecord(
        EvaluationSource(
            Protocol13Benchmark(source["benchmark"]),
            source["population_id"],
            source["source_id"],
            source["episode_id"],
        ),
        RolloutTask.from_value(value["public_task"]),
        EvaluationTarget(value["native_target"]),
    )


def freeze_iid_architecture(
    *,
    destination: Path,
    architecture_id: str,
    source: SourcePanel,
    identities: tuple[IIDSourceIdentity, ...],
    excluded_sources: dict[str, frozenset[tuple[str, str]]],
    formal: BayesianFormalConfig,
    initial_library: SkillLibraryState,
    model_controls: dict[str, Any],
    scorer_controls: dict[str, Any],
    environment_controls: dict[str, Any],
    serving_controls: dict[str, Any],
    workflow: RolloutWorkflowBinding,
    acceptance_rules: dict[str, Any],
    prior_evaluation_sources: frozenset[tuple[str, str]] = frozenset(),
    source_aliases: dict[str, dict[str, str]] | None = None,
) -> Path:
    """Freeze both baseline arms before generation, without constructing a trainer.

    Split declarations describe project exposure, not pretraining non-exposure.
    Historical evaluation exposure is retained separately from this candidate's
    training/development/quality exclusions. It never grants an overlap exception
    for those exclusions, nor makes this acceptance reference an unseen final test.
    Thresholds must already have been declared using other/development evidence.
    A failed final panel is retained; there is no score-driven refreeze here.
    """
    require_fresh_interface(formal)
    source.panel.validate(canary=False)
    if not architecture_id.strip() or {e.benchmark for e in source.panel.entries} != set(
        formal.domains
    ):
        raise ValueError("architecture-matched IID must match the declared active domains")
    if set(excluded_sources) != {"training", "development", "quality"}:
        raise ValueError("declare training, development and quality source exclusions separately")
    source_aliases = source_aliases or {}
    if any(
        domain not in {d.value for d in SEVEN_TRAINING_DOMAINS}
        or not isinstance(mapping, dict)
        or any(
            not isinstance(original, str)
            or not original
            or not isinstance(canonical, str)
            or not canonical
            for original, canonical in mapping.items()
        )
        for domain, mapping in source_aliases.items()
    ):
        raise ValueError("native source aliases must declare nonempty original and canonical IDs")
    for excluded in excluded_sources.values():
        require_source_isolation(source.panel.entries, identities, excluded, source_aliases)
    prior_overlap = sorted(
        {canonical_source_key(row.canonical_source, source_aliases) for row in identities}
        & {canonical_source_key(row, source_aliases) for row in prior_evaluation_sources}
    )
    if set(scorer_controls) != set(formal.domains):
        raise ValueError("freeze the native scorer for every active domain")
    if scorer_controls["healthbench"] != healthbench_condition(formal.healthbench_judge):
        raise ValueError("IID HealthBench judge must match the candidate training condition")
    if not all((model_controls, serving_controls, environment_controls)):
        raise ValueError("model, serving and native environment controls must be expanded")
    if set(acceptance_rules) != set(formal.domains) or any(
        not isinstance(rule, dict)
        or not isinstance(rule.get("metric"), str)
        or not rule["metric"]
        or type(rule.get("minimum")) not in (int, float)
        or not math.isfinite(rule["minimum"])
        or rule.get("comparison", "at-least") not in {"at-least", "strictly-above"}
        for rule in acceptance_rules.values()
    ):
        raise ValueError("declare a native metric and acceptance floor per domain before A0")
    records = evaluation_records(source, identities, max_turns=formal.max_turns, seed=formal.seed)
    destination.mkdir(parents=True, mode=0o700, exist_ok=False)
    path = destination / "architecture-private.json"
    durable_json(
        path,
        normalize_json(
            {
                "format": FORMAT,
                "architecture_id": architecture_id,
                "isolation": evaluation_isolation(),
                "arms": ARMS,
                "panel": {
                    "purpose": "architecture-acceptance-reference",
                    "independent_unseen_final_test": False,
                    "exposure": source.panel.exposure,
                    "prior_project_evaluation_sources": sorted(prior_evaluation_sources),
                    "prior_project_evaluation_overlap": prior_overlap,
                    "source_aliases": source_aliases,
                    "source_provenance": source.provenance,
                    "identities": [asdict(row) for row in identities],
                    "records": [record_value(record) for record in records],
                    "public_inputs": [
                        {
                            "task_id": entry.task_id,
                            "input_profile": entry.input_profile,
                            "fields": dict(entry.fields),
                            "source_paragraphs": source.public_input_receipts.get(entry.task_id),
                        }
                        for entry in source.panel.entries
                    ],
                    "excluded_sources": {k: sorted(v) for k, v in excluded_sources.items()},
                },
                "controls": {
                    "execution_machine": "skillev.rollout.episode_executor.execute_episode@1",
                    "formal": formal.expanded_value(),
                    "sampling": formal.sampling_config.to_value(),
                    "domain_budgets": {k: asdict(v) for k, v in formal.domain_task_budgets.items()},
                    "model": model_controls,
                    "scorers": scorer_controls,
                    "environments": environment_controls,
                    "serving": serving_controls,
                    "workflow": workflow.to_value(),
                    "actor_transport": {
                        key: value
                        for key, value in formal_actor_transport(
                            "http://unused.invalid",
                            worker_threads=workflow.transport_worker_threads,
                        )
                        .to_value()
                        .items()
                        if key != "endpoint_base"
                    },
                    "initial_context_profile": "trained-skillev@1",
                    "sampling_schedule": "architecture-matched-iid-seed0@1",
                    "sampling_anchor": 0,
                    "maximum_h0_tokens": formal.max_input_tokens,
                },
                "initial_library": initial_library.to_value(),
                "acceptance_rules": acceptance_rules,
            }
        ),
    )
    return path


@dataclass(frozen=True)
class IIDSnapshotSelection:
    arm: str
    policy: PolicySnapshot
    optimizer_step: int = 0
    library: SkillLibraryState | None = None

    def __post_init__(self) -> None:
        if self.arm not in ARMS or type(self.optimizer_step) is not int or self.optimizer_step < 0:
            raise ValueError("invalid declared IID snapshot axis")
        if self.arm == "skills-off" and self.library is not None:
            raise ValueError("skills-off cannot acquire a library through snapshot substitution")


def expanded_controls(
    architecture: dict[str, Any],
    selection: IIDSnapshotSelection,
) -> dict[str, Any]:
    if architecture["format"] != FORMAT:
        raise ValueError("not an architecture-matched IID bundle")
    library = (
        SkillLibraryState.from_seed_documents(())
        if selection.arm == "skills-off"
        else selection.library or SkillLibraryState.from_value(architecture["initial_library"])
    )
    return cast(
        dict[str, Any],
        _json(
            {
                "format": FORMAT,
                "architecture_id": architecture["architecture_id"],
                "arm": selection.arm,
                "controls": copy.deepcopy(architecture["controls"]),
                "panel": copy.deepcopy(architecture["panel"]),
                "acceptance_rules": copy.deepcopy(architecture["acceptance_rules"]),
                "policy_snapshot": selection.policy.to_value(),
                "optimizer_step": selection.optimizer_step,
                "library_snapshot": library.to_value(),
                "isolation": evaluation_isolation(),
            }
        ),
    )


def require_iid_architecture_match(
    reference: dict[str, Any],
    current: dict[str, Any],
    *,
    allow_library_change: bool = False,
) -> tuple[str, ...]:
    """Only explicit weights/library interventions; compare actual expanded controls."""
    left, right = _json(reference), _json(current)
    axes = []
    for key in ("policy_snapshot", "optimizer_step"):
        if left.pop(key) != right.pop(key) and "policy" not in axes:
            axes.append("policy")
    old_library, new_library = left.pop("library_snapshot"), right.pop("library_snapshot")
    if old_library != new_library:
        if not allow_library_change or reference["arm"] != "initial-library":
            raise ValueError("library intervention was not declared for this comparison")
        axes.append("library")
    if left != right:
        raise ValueError("non-snapshot IID execution, data, scoring or budget controls changed")
    if (reference["policy_snapshot"] == current["policy_snapshot"]) != (
        reference["optimizer_step"] == current["optimizer_step"]
    ):
        raise ValueError("policy step and snapshot intervention disagree")
    return tuple(axes)
