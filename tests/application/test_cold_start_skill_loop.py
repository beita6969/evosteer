"""Controlled execution integration, NOT natural Qwen efficacy evidence."""

import asyncio
from dataclasses import replace

from skillev.contracts import EvolutionActionType
from skillev.diagnostics.rollout_progress import RolloutProgress, bind_progress
from skillev.evolution import (
    AuthoredSkillDraft,
    AuthoringResult,
    FullEvolutionPolicy,
    SkillAuthoringAuthority,
    TaskConditionedSkillRetriever,
    build_evolution_mutation,
)
from skillev.evolution.authoring import uncovered_edge_requirement_id
from skillev.evolution.evidence import (
    TrajectoryEvidenceView,
    WindowFlowView,
    assemble_posterior_evidence_view,
)
from skillev.rollout import CanonicalInitialContextAssembler, DecodingSnapshot
from skillev.rollout.catalog import CatalogReadEnvironment
from skillev.runtime import SkillApplicability, SkillLibrary, SkillLibraryState, SkillRequirement
from skillev.training.coverage_reporting import batch_coverage
from skillev.training.evidence_context import TrajectoryEvidenceContext
from skillev.training.skill_lifecycle import skill_lifecycle_report
from tests.evolution.test_cold_start import config, detect, prepared_window
from tests.fakes.skill_author import ScriptedSkillAuthor
from tests.rollout.engine_fakes import (
    ByteTokenizer,
    GenerationScript,
    ScriptedEnvironment,
    default_request,
    make_harness,
)
from tests.rollout.test_native_tool_wire import call, contract
from tests.training.test_bayesian_chain import restore
from tests.v3_helpers import CharacterTokenizer, make_skill_document, make_source


def test_generated_skill_enters_next_catalog_body_and_posterior_only_after_actual_read(tmp_path):
    library = SkillLibrary(SkillLibraryState.from_seed_documents((make_skill_document("seed"),)))
    projection = prepared_window(library_version=library.current_version)
    _, detected = detect(projection, config())
    frozen = projection.freeze_for_phase(detected.event)
    scope = next(iter(frozen.authoring_by_edge_id.values()))
    decision = FullEvolutionPolicy().decide_from_views(
        window_flow=WindowFlowView(
            detected.event,
            frozen.diagnostics,
            library.active_skill_ids,
            {s: library.document(s).applicability for s in library.active_skill_ids},
            (scope.task_family,),
        ),
        posterior=assemble_posterior_evidence_view(
            active_skill_ids=library.active_skill_ids, cells=(), event_ids_by_cell={}
        ),
        trajectories=TrajectoryEvidenceView(
            frozen.authoring_by_edge_id, frozen.source_by_trajectory
        ),
        config=config(),
    )
    draft = AuthoredSkillDraft(
        title="Check a public result before submitting",
        summary="For this task family, verify the public operation result before completion.",
        instructions="NEW_BODY: inspect the public result; check the output; then submit.",
        applicability=SkillApplicability(
            (scope.task_family,), (scope.context_id,), scope.available_tools, ()
        ),
        requirements=tuple(
            SkillRequirement(
                uncovered_edge_requirement_id(e.edge_id),
                "Check the public execution result before submitting.",
                kind="evolvable-strategy",
            )
            for e in decision.proposals[0].edge_exemplars
        ),
    )
    author = ScriptedSkillAuthor(CharacterTokenizer(), (AuthoringResult((draft,)),))
    mutation = build_evolution_mutation(
        decision,
        author=author,
        library=library,
        phase_event=detected.event,
        config=config(),
        authority=SkillAuthoringAuthority(
            "input@3", "output@3", "unit-test", (scope.task_family,), scope.available_tools
        ),
        base_seed=0,
        cycle_ordinal=1,
    )
    assert mutation.actions[0].action_type is EvolutionActionType.GENERATE
    assert len(author.requests) == 1
    before = library.current_version
    # Keep the original retriever object: it must not retain stale documents.
    retriever = TaskConditionedSkillRetriever(library=library)
    request = default_request()
    task = replace(
        request.task,
        task_family=scope.task_family,
        context_id=scope.context_id,
        available_tools=scope.available_tools,
        action_surface=contract().surface,
    )
    old_ids = {s.metadata.skill_id for s in retriever.retrieve(task)}
    library.apply(library.preview(mutation))
    projection.commit_reset(
        projection.preview_reset(old_version=before, new_version=library.current_version)
    )
    new_id = mutation.new_documents[0].manifest.skill_id
    assert new_id not in old_ids
    assert projection.calibration_cells == ()
    retrieved = retriever.retrieve(task)
    assert new_id in {s.metadata.skill_id for s in retrieved}
    assert new_id not in {
        s.metadata.skill_id
        for s in retriever.retrieve(replace(task, task_family="unrelated", context_id="unrelated"))
    }
    tokenizer = ByteTokenizer()
    request = replace(
        request,
        task=task,
        retrieved_skills=retrieved,
        active_skill_ids=library.active_skill_ids,
        library_version=library.current_version,
        decoding=DecodingSnapshot.create(
            max_reasoning_tokens=128,
            max_action_tokens=4096,
            base_seed=17,
            action_boundary_version="native-model-stop@1",
        ),
    )
    harness = make_harness(
        tmp_path,
        tokenizer=tokenizer,
        request=request,
        max_turns=2,
        context_assembler=CanonicalInitialContextAssembler(
            maximum_h0_tokens=20000,
            phase_context=True,
            action_wire="native-single-tool-call@3",
            skill_exposure="catalog-then-read@1",
        ),
        environment=CatalogReadEnvironment(
            ScriptedEnvironment([], task_family=scope.task_family),
            retrieved,
            library.current_version,
        ),
        scripts=[
            GenerationScript.text(tokenizer, text)
            for text in (
                "Use one relevant procedure.",
                call("read_skill", skill_id=new_id),
                "Apply the returned guidance.",
                call("submit_answer", answer="synthetic"),
            )
        ],
    )
    progress = RolloutProgress(trajectory_id=request.trajectory_id)
    with bind_progress(progress):
        artifact = asyncio.run(harness.engine.run(request))
    assert new_id in progress.snapshot()["phases"][0]["catalog_visible_skill_ids"]
    prompts = [tokenizer.decode(r.input_ids) for r in harness.generator.requests]
    assert "NEW_BODY" not in prompts[0]
    assert draft.summary in prompts[0]
    assert "NEW_BODY" in prompts[2]
    assert "at least two" not in prompts[0]
    assert artifact.record.horizon == 2  # one optional read, one task action
    context = TrajectoryEvidenceContext.from_artifact(artifact)
    assert context.invocation_links[0].body_returned is True
    assert context.invocation_links[0].following_execution_steps == (2,)
    assert context.body_visible_skill_ids == (new_id,)
    # Fixture score coordinates are explicitly bound to the new actual rollout.
    source = make_source(
        batch_id="new-library-batch",
        optimizer_step=3,
        library_version=library.current_version,
        artifacts=(artifact,),
    )
    projection.commit(projection.preview(source))
    update = projection.posterior_provenance.batches[-1].posterior.updates[0]
    assert update.skill_id == new_id
    assert (update.alpha_before, update.beta_count_before) == (1, 1)
    assert update.outcome == artifact.record.reward.success
    assert all(not b.posterior.updates for b in projection.posterior_provenance.batches[:2])
    restored = restore(projection)
    library = SkillLibrary(SkillLibraryState.from_value(library.state.to_value()))
    recovered_retriever = TaskConditionedSkillRetriever(library=library)
    assert new_id in {s.metadata.skill_id for s in recovered_retriever.retrieve(task)}
    recovered_environment = CatalogReadEnvironment(
        ScriptedEnvironment([], task_family=scope.task_family),
        recovered_retriever.retrieve(task),
        library.current_version,
    )
    from skillev.rollout.codec import codec_for_initial_meta

    action = (
        codec_for_initial_meta(artifact.record.initial_context.meta)
        .parse(call("read_skill", skill_id=new_id))
        .action
    )
    assert asyncio.run(recovered_environment.execute(action, step_index=1)).public_value[
        "content"
    ] == next(
        s.content for s in recovered_retriever.retrieve(task) if s.metadata.skill_id == new_id
    )
    assert restored.calibration_cells == projection.calibration_cells
    report = skill_lifecycle_report(library, restored.posterior_provenance)
    new_skill = next(item for item in report["skills"] if item["skill_id"] == new_id)
    assert new_skill["batches"][0]["optimizer_step"] == 3
    assert new_skill["batches"][0]["body_read_count"] == 1
    assert new_skill["batches"][0]["reads_visible_in_later_action"] == 1
    assert report["catalogs"][-1]["input_evidence"] == list(artifact.skill_input_evidence)
    assert new_skill["evidence_status"] == "actual-invocation-observed"
    coverage = batch_coverage(restored.posterior_provenance.batches[-1])
    assert coverage["benchmarks"]["unknown"]["body_visible_skill_ids"] == [new_id]
