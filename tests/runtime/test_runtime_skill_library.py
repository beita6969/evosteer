from __future__ import annotations

from dataclasses import replace

import pytest

from skillev.contracts import PruneActionRecord, PruneEvidence, stable_hash
from skillev.evolution import EvolutionMutation, TaskConditionedSkillRetriever
from skillev.rollout import CanonicalInitialContextAssembler, RolloutTask
from skillev.runtime import (
    EMPTY_LIBRARY_VERSION,
    SkillLibrary,
    SkillLibraryState,
)
from skillev.runtime.skill_library import skill_library_version
from tests.v3_helpers import make_skill_document


class _Tokenizer:
    tokenizer_id = "library-wire-tokenizer@1"

    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]


def test_library_state_is_content_addressed_and_round_trips() -> None:
    alpha = make_skill_document("alpha")
    beta = make_skill_document("beta")
    state = SkillLibraryState.from_seed_documents((beta, alpha))

    assert state.active_skill_ids == ("skill-alpha", "skill-beta")
    assert SkillLibraryState.from_value(state.to_value()) == state
    assert state.current_version != EMPTY_LIBRARY_VERSION


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", "2"),
        ("input_schema_id", "alternate-input@1"),
        ("output_schema_id", "alternate-output@1"),
        ("license_id", "alternate-license"),
        ("provenance_hash", stable_hash({"provenance": "alternate"})),
    ],
)
def test_active_library_version_commits_every_manifest_wire_field(
    field: str,
    value: str,
) -> None:
    original = make_skill_document("alpha")
    mutated = replace(original, manifest=replace(original.manifest, **{field: value}))

    original_state = SkillLibraryState.from_seed_documents((original,))
    mutated_state = SkillLibraryState.from_seed_documents((mutated,))

    assert original.manifest.content_hash == mutated.manifest.content_hash
    assert original_state.current_version != mutated_state.current_version
    assert original_state.state_hash != mutated_state.state_hash


def test_library_state_hash_commits_inactive_lineage_and_document_format_is_closed() -> None:
    active = make_skill_document("alpha")
    inactive = make_skill_document("beta")
    active_only = SkillLibraryState.from_seed_documents((active,))
    with_inactive = SkillLibraryState(
        documents={active.manifest.skill_id: active, inactive.manifest.skill_id: inactive},
        active_skill_ids=(active.manifest.skill_id,),
        current_version=skill_library_version(
            documents={active.manifest.skill_id: active, inactive.manifest.skill_id: inactive},
            active_skill_ids=(active.manifest.skill_id,),
        ),
    )
    malformed_document = active.to_value()
    malformed_document["format"] = "skillev-skill-document@future"

    assert active_only.current_version == with_inactive.current_version
    assert active_only.state_hash != with_inactive.state_hash
    with pytest.raises(ValueError):
        type(active).from_value(malformed_document)


def test_same_active_library_identity_reconstructs_the_same_h0() -> None:
    document = make_skill_document("alpha")
    first = SkillLibraryState.from_seed_documents((document,))
    second = SkillLibraryState.from_value(first.to_value())
    task = RolloutTask(
        task_id="library-wire-task",
        environment_id="library-wire-environment",
        task_family="debug/task-family",
        context_id="debug:context-id",
        query="Use the active public skill.",
        available_tools=(),
        public_context={"debug": True},
    )
    assembler = CanonicalInitialContextAssembler(maximum_h0_tokens=4_096)
    tokenizer = _Tokenizer()

    def assemble(state: SkillLibraryState):
        library = SkillLibrary(state)
        return assembler.assemble(
            task=task,
            retrieved_skills=TaskConditionedSkillRetriever(library=library).retrieve(task),
            active_skill_ids=library.active_skill_ids,
            library_version=library.current_version,
            tokenizer=tokenizer,
        )

    first_h0 = assemble(first)
    second_h0 = assemble(second)

    assert first.current_version == second.current_version
    assert first_h0.text == second_h0.text
    assert first_h0.contract.assembled_hash == second_h0.contract.assembled_hash


def test_preview_is_pure_and_apply_publishes_one_immutable_state() -> None:
    alpha = make_skill_document("alpha")
    state = SkillLibraryState.from_seed_documents((alpha,))
    library = SkillLibrary(state)
    after_version = EMPTY_LIBRARY_VERSION
    action = PruneActionRecord(
        action_id=stable_hash({"action": "prune-alpha"}),
        phase_event_id=stable_hash({"phase": "one"}),
        library_version_before=state.current_version,
        library_version_after=after_version,
        target_skill_id=alpha.manifest.skill_id,
        lineage_ref=stable_hash({"lineage": "one"}),
        proposal_content_hash=stable_hash({"proposal": "prune-alpha"}),
        evidence=PruneEvidence.observed_low(
            log_skill_marginal_flow=-2.0,
            flow_quantile=0.1,
            ucb=0.2,
            k=1.0,
            posterior_event_ids=("posterior-1",),
        ),
        rationale_text="low flow and low optimistic posterior",
    )
    mutation = EvolutionMutation(
        library_version_before=state.current_version,
        library_version_after=after_version,
        new_documents=(),
        active_skill_ids_after=(),
        actions=(action,),
    )

    preview = library.preview(mutation)

    assert library.state is state
    assert preview.current_version == EMPTY_LIBRARY_VERSION
    assert preview.active_skill_ids == ()
    library.apply(preview)
    assert library.state is preview
    assert library.all_documents() == (alpha,)


def test_library_owns_no_event_log_or_history_rebuilder() -> None:
    library = SkillLibrary(SkillLibraryState.from_seed_documents(()))

    assert not hasattr(library, "emitter")
    assert not hasattr(library, "from_events")
    assert not hasattr(library, "rebuild")
    with pytest.raises(KeyError):
        library.document("missing")
