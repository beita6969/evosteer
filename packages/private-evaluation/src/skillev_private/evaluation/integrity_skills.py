"""Read only the selected checkpoint's skill projection, never restore a trainer."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from skillev.evaluation.skill_library_config import FrozenSkillLibrary, initial_skill_library
from skillev.runtime import SkillLibraryState
from skillev.runtime.runtime_snapshot import RUNTIME_SNAPSHOT_FORMAT


def read_skill_library(library_id: str, settings: dict[str, Any]) -> FrozenSkillLibrary:
    kind, rule = settings["kind"], settings["retrieval_rule"]
    if kind == "initial" and settings.get("source") == "planned-advisory-seeds@1":
        return replace(initial_skill_library(), library_id=library_id, retrieval_rule=rule)
    if kind == "initial":
        path = Path(settings["snapshot_file"]).expanduser()
        if not path.is_absolute():
            raise ValueError("skill snapshot must be an explicitly selected absolute file")
        state = SkillLibraryState.from_value(json.loads(path.read_text(encoding="utf-8")))
        return FrozenSkillLibrary(library_id, kind, state, retrieval_rule=rule)
    if kind != "evolved":
        raise ValueError("unknown skill library source kind")
    path = Path(settings["checkpoint_directory"]).expanduser()
    if not path.is_absolute():
        raise ValueError("skill checkpoint must be an explicitly selected absolute directory")
    if not (path / "COMPLETE").is_file():
        raise ValueError("skill checkpoint has not finished writing")
    metadata = json.loads((path / "runtime_state.json").read_text(encoding="utf-8"))
    if metadata.get("format") != RUNTIME_SNAPSHOT_FORMAT:
        raise ValueError("unsupported skill checkpoint format")
    # Use the existing canonical library decoder, not RuntimeSnapshot.restore:
    # the latter reconstructs training diagnostics and requires optimizer state.
    state = SkillLibraryState.from_value(metadata["execution_state"]["library"])
    execution = metadata["execution_state"]
    cursor = execution.get("run_cursor", {})
    provenance = execution.get("projections", {}).get("posterior_provenance")
    return FrozenSkillLibrary(
        library_id,
        kind,
        state,
        metadata["optimizer_step"],
        rule,
        source_checkpoint=str(path.resolve()),
        actual_mutation_count=cursor.get("committed_cycles"),
        committed_proposal_count=cursor.get("committed_actions"),
        posterior_update_count=None
        if provenance is None
        else sum(len(batch["posterior"]["updates"]) for batch in provenance["batches"]),
        initial_library_version=metadata.get("identity", {}).get("initial_library_version"),
    )
