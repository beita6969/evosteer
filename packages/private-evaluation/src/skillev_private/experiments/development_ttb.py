"""Read-only real TTB checkpoint binding and complete source exclusions.

Reads existing COMPLETE/runtime/policy metadata and original source records. No
tensor loading, hashing, optimizer construction, publication, or state writes.
The independently published actor adapter uses the existing versioned gateway.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from skillev.policy.checkpoint import read_policy_checkpoint_state
from skillev.rollout import PolicySnapshot
from skillev.runtime import SkillDocument, SkillLibraryState
from skillev.runtime.execution_state import FullRuntimeExecutionState
from skillev.runtime.step_transaction import StepTransactionRecord, StepTransactionState
from skillev.training.checkpoint import FilesystemTrainingCheckpointStore
from skillev.training.run_condition import EffectiveRunCondition
from skillev_private.evaluation.iid_episode_sources import canonical_source_key

FORMAT = "development-ttb-checkpoint@1"


def _read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sources(rows: Any, aliases: Any) -> set[tuple[str, str]]:
    if not isinstance(rows, list):
        raise ValueError("original source inventory must contain source rows")
    result = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("source inventory rows must be structured coordinates")
        domain = row.get("benchmark", row.get("benchmark_id"))
        source = row.get("source_id", row.get("source_question_id"))
        if not isinstance(domain, str) or not isinstance(source, str) or not source.strip():
            raise ValueError("original source coordinates are missing")
        result.add(canonical_source_key((domain, source), aliases))
    return result


def _skill_sources(
    path: Path,
    *,
    initial: SkillLibraryState,
    current: SkillLibraryState,
    aliases: Any,
    training_sources: set[tuple[str, str]],
    optimizer_step: int,
) -> tuple[set[tuple[str, str]], dict[str, Any]]:
    """Every stored document, including inactive lineage, has a source account.

    An initial public procedure may declare no task sources. A new Phi document
    must be present in its saved mutation, not merely labelled as evolved. Any
    additional construction/development sources come from archived source files.
    Excluding all training sources also covers unsuccessful/uninvoked Phi edges.
    """
    manifest = _read(path)
    if manifest.get("format") != "skill-construction-sources@1":
        raise ValueError("explicit initial and evolved skill source inventory is required")
    entries = manifest.get("documents")
    expected = {**initial.documents, **current.documents}
    if not isinstance(entries, list) or len(entries) != len(expected):
        raise ValueError("skill source inventory must cover every initial/evolved document")
    seen = set()
    sources = set()
    evidence = []
    for entry in entries:
        document = SkillDocument.from_value(entry["document"])
        identity = document.manifest.skill_id
        if identity in seen or expected.get(identity) != document:
            raise ValueError("skill source inventory differs from the actual saved library")
        seen.add(identity)
        kind = entry.get("kind")
        files = entry.get("source_files")
        if not isinstance(files, list):
            raise ValueError("each skill must explicitly list its construction source files")
        originals = []
        for filename in files:
            raw = _read(filename)
            rows = raw["ordered_sources"] if isinstance(raw, dict) else raw
            keys = _sources(rows, aliases)
            if not keys:
                raise ValueError("empty source files are not construction provenance")
            sources.update(keys)
            originals.append({"path": filename, "sources": [list(key) for key in sorted(keys)]})
        mutation = None
        if kind == "public-procedure":
            if initial.documents.get(identity) != document:
                raise ValueError("an evolved skill cannot be relabelled an initial public seed")
            if not isinstance(entry.get("public_basis"), str) or not entry["public_basis"].strip():
                raise ValueError("public procedures require an explicit public construction basis")
        elif kind == "source-derived":
            if not originals:
                raise ValueError("source-derived skill needs the archived construction sources")
        elif kind == "training-evolution":
            mutation = _read(entry["mutation_record"])
            # Accept a saved source-event envelope or its original payload.
            payload = mutation.get("payload", mutation)
            step = payload.get("optimizer_step")
            if type(step) is not int or not 1 <= step <= optimizer_step:
                raise ValueError("skill mutation is outside this checkpoint's committed prefix")
            generated = payload.get("mutation", {}).get("new_documents", [])
            if document.to_value() not in generated:
                raise ValueError("actual mutation record did not generate this skill body")
            sources.update(training_sources)
        else:
            raise ValueError("unknown skill construction source kind")
        evidence.append(
            {
                "document": document.to_value(),
                **entry,
                "source_records": originals,
                "mutation_payload": mutation,
            }
        )
    if seen != set(expected):
        raise ValueError("skill source inventory omits saved library lineage")
    return sources, {"manifest": manifest, "resolved_documents": evidence}


def resolve_ttb(binding: dict[str, Any]) -> tuple[PolicySnapshot, dict[str, Any]]:
    from .autonomous_ttb import require_autonomous_initialization
    from .bayesian_training_config import BayesianFormalConfig
    from .bayesian_training_setup import _read_preparation
    from .warmup_initialization import initialization_condition

    if binding.get("format") != FORMAT or binding.get("kind") != "ttb-checkpoint":
        raise ValueError("real TTB binding cannot impersonate a supervised initialization")
    for name in (
        "checkpoint",
        "run_root",
        "preparation",
        "effective_condition",
        "skill_sources",
        "published_forward_adapter",
    ):
        if not isinstance(binding.get(name), str) or not binding[name].strip():
            raise ValueError("TTB binding requires exact checkpoint, original run and source files")
    root = Path(binding["run_root"]).resolve()
    path = Path(binding["checkpoint"]).resolve()
    if path.parent != root / "checkpoints":
        raise ValueError("select a complete checkpoint from this run, not a detached F export")
    metadata = FilesystemTrainingCheckpointStore(root=path.parent).load_metadata(path)
    state = metadata.execution_state
    if metadata.experiment_id != root.name or not isinstance(state, FullRuntimeExecutionState):
        raise ValueError("candidate must be this run's complete TTB/Bayesian checkpoint")
    step = metadata.optimizer_step
    if step < 1 or state.run_cursor.completed_training_steps != step:
        raise ValueError("TTB candidate needs real committed updates, not renamed Step-0")
    config = BayesianFormalConfig.load(root / "formal-config.json")
    preparation = Path(binding["preparation"])
    backbone, original = _read_preparation(preparation)
    require_autonomous_initialization(config, preparation)
    if initialization_condition(preparation):
        raise ValueError("this pure-TTB comparison starts before specialized skill SFT")
    initial_policy = read_policy_checkpoint_state(Path(original.directory))
    if (
        initial_policy.optimizer_step != 0
        or initial_policy.trainable_state != original.trainable_state
    ):
        raise ValueError("before snapshot must be the saved initialization used by this TTB run")
    actual_policy = read_policy_checkpoint_state(path / metadata.policy_directory)
    if (
        actual_policy.optimizer_step != step
        or actual_policy.backbone_id != initial_policy.backbone_id
        or metadata.identity.initial_trainable_state_hash != original.trainable_state.content_hash
        or state.task_cursor.cursor != step * config.batch_size
    ):
        raise ValueError("runtime cursor, original initialization and actual policy disagree")
    condition_path = Path(binding["effective_condition"]).resolve()
    if condition_path.parent != root or not condition_path.name.startswith(
        "effective-condition-process-"
    ):
        raise ValueError("read the original same-run expanded condition")
    condition = EffectiveRunCondition.from_value(_read(condition_path))
    science = cast(dict[str, Any], condition.scientific)
    if science.get("initial_partition") != original.trainable_state.to_value():
        raise ValueError("source run's original partition differs from the before policy")
    declared_formal = science["formal"]
    if any(
        declared_formal.get(k) != v
        for k, v in config.to_value().items()
        if k
        not in {
            "performance_profile",
            "planning_hours",
            "target_steps_per_hour",
            "checkpoint_every",
        }
    ):
        raise ValueError("source run's original method controls changed")
    data = science["data_condition"]
    rows = data["ordered_selected_sources"]
    if len(rows) != config.steps * config.batch_size or len(
        {r["occurrence_id"] for r in rows}
    ) != len(rows):
        raise ValueError("exclude the complete planned training schedule, not only invoked sources")
    declared_sources = data.get("declaration", {})
    aliases = declared_sources.get("autonomous_ttb_sources", declared_sources).get(
        "source_aliases", {}
    )
    training_sources = _sources(rows, aliases)
    initial_library = SkillLibraryState.from_value(science["initial_library"])
    if initial_library.current_version != metadata.identity.initial_library_version:
        raise ValueError("initial library differs from the source training checkpoint")
    skill_sources, inventory = _skill_sources(
        Path(binding["skill_sources"]),
        initial=initial_library,
        current=state.library,
        aliases=aliases,
        training_sources=training_sources,
        optimizer_step=step,
    )
    policy = PolicySnapshot.create(
        backbone_id=actual_policy.backbone_id,
        forward_adapter_version=actual_policy.forward_version,
        tokenizer_id=backbone.tokenizer_id,
        backend_id="sglang-native-exact-token",
        initial_trainable_state_hash=metadata.identity.initial_trainable_state_hash,
    )
    transaction = StepTransactionRecord.from_value(
        _read(path.parent / "step-transactions" / f"step-{step:08d}.json")
    )
    if (
        transaction.state is not StepTransactionState.COMMITTED
        or transaction.optimizer_step != step
        or transaction.policy_snapshot_after != policy.snapshot_id
        or transaction.adapter_revision != actual_policy.forward_version
    ):
        raise ValueError("COMPLETE files alone do not prove a committed training transaction")
    before = PolicySnapshot.create(
        backbone_id=initial_policy.backbone_id,
        forward_adapter_version=initial_policy.forward_version,
        tokenizer_id=backbone.tokenizer_id,
        backend_id=policy.backend_id,
        initial_trainable_state_hash=metadata.identity.initial_trainable_state_hash,
    )
    return policy, {
        **binding,
        "backbone": backbone.to_value(),
        "policy": policy.to_value(),
        "before_policy": before.to_value(),
        "checkpoint_state": actual_policy.to_value(),
        "source_optimizer_step": step,
        "source_run": metadata.experiment_id,
        "sampling": config.sampling_config.to_value(),
        "training_sources": [list(key) for key in sorted(training_sources)],
        "skill_construction_sources": [list(key) for key in sorted(skill_sources)],
        "source_aliases": aliases,
        "source_inventory": inventory,
        "initial_library": initial_library.to_value(),
        "checkpoint_library": state.library.to_value(),
        "source_committed_cycles": state.run_cursor.committed_cycles,
        "original_step_zero": False,
        "collection_pairing_step": 0,
        "collection_pairing_is_not_training_step": True,
        "model_loaded_by_collector": False,
        "adapter_publication": "owner-prepublished-existing-gateway-binding",
        "training_state_writes": False,
    }
