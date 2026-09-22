"""Real tiny CPU LoRA/Adam lifecycle; all examples/reviews are synthetic fixtures."""

import asyncio
import json
from dataclasses import replace

import pytest
import torch
from skillev_private.experiments import prepare_fresh_restart as prepare
from skillev_private.experiments import skill_use_warmup as entry
from skillev_private.experiments.bayesian_training_setup import _read_preparation
from skillev_private.experiments.fresh_restart import require_clean_initial_application
from skillev_private.experiments.warmup_initialization import (
    WARMUP_PREPARATION_FORMAT,
    initialization_condition,
    require_initialization_candidate,
)

from skillev.contracts import canonical_json
from skillev.experiments import _evolution_preflight_seed
from skillev.policy import QwenBackboneConfig, QwenMultimodalBackboneConfig, QwenPolicyBackbone
from skillev.rollout import CanonicalInitialContextAssembler, DecodingSnapshot
from skillev.rollout.catalog import CatalogReadEnvironment
from skillev.runtime import FullRetrievedSkillContext, RetrievalInclusionReason, SkillMetadata
from skillev.scoring.edge_plan import prepare_edge_plan
from skillev.training.skill_use_warmup import WarmupConfig
from tests.application.test_full_vertical_loop import build_application_fixture
from tests.rollout.engine_fakes import (
    FakeTerminalEvaluator,
    GenerationScript,
    ScriptedEnvironment,
    default_request,
    make_harness,
)
from tests.rollout.test_native_tool_wire import call, contract


def write(path, value):
    path.write_text(json.dumps(value))
    return path


def corpus_files(root, backbone, library, review_kind):
    root.mkdir()
    doc = next(iter(library.documents.values()))
    skill = FullRetrievedSkillContext(
        SkillMetadata.from_document(doc),
        canonical_json(doc.content_value()),
        RetrievalInclusionReason.APPLICABILITY_MATCH,
    )
    demos = []
    for name, decision in (
        ("applied", "applied"),
        ("applied-second", "applied"),
        ("not-called", "not-called"),
    ):
        directory = root / name
        directory.mkdir()
        request = default_request(trajectory_id=name)
        task = replace(request.task, task_id=name, action_surface=contract().surface)
        request = replace(
            request,
            task=task,
            sampling_coordinate=replace(request.sampling_coordinate, task_id=name),
            library_version=library.current_version,
            retrieved_skills=(skill,),
            active_skill_ids=(skill.metadata.skill_id,),
            decoding=DecodingSnapshot.create(
                max_reasoning_tokens=128,
                max_action_tokens=4096,
                base_seed=0,
                action_boundary_version="native-model-stop@1",
            ),
        )
        texts = ["finish", call("submit_answer", answer="synthetic")]
        if decision == "applied":
            texts = ["consider", call("read_skill", skill_id=skill.metadata.skill_id), *texts]
        harness = make_harness(
            directory,
            tokenizer=backbone.tokenizer,
            request=request,
            max_turns=2,
            evaluator=FakeTerminalEvaluator(value=1),
            context_assembler=CanonicalInitialContextAssembler(
                maximum_h0_tokens=20000,
                phase_context=True,
                action_wire="native-single-tool-call@3",
                skill_exposure="catalog-then-read@1",
            ),
            environment=CatalogReadEnvironment(
                ScriptedEnvironment([]), (skill,), library.current_version
            ),
            scripts=[
                GenerationScript.text(backbone.tokenizer, text, stop_token_ids=()) for text in texts
            ],
        )
        artifact = asyncio.run(harness.engine.run(request))
        plan = prepare_edge_plan(backbone.tokenizer, artifact.record, artifact.initial_context.text)
        note = {
            "format": "skill-application-annotation@2",
            "review_kind": review_kind,
            "reviewer": "synthetic-reviewer-not-a-real-endorsement",
            "reviewed_at": "synthetic-date",
            "approved_for_warmup": True,
            "trajectory_id": name,
            "rationale": "Synthetic fixture for explicit application attribution, not efficacy.",
            "decision": decision,
            "skill_id": skill.metadata.skill_id if decision == "applied" else None,
            "read_step": 1 if decision == "applied" else None,
            "decision_step": 2 if decision == "applied" else None,
            "behavior_evidence": {
                "method_clauses": [
                    {"clause_id": "procedure", "quote": doc.content_value()["instructions"]}
                ]
                if decision == "applied"
                else [],
                "actions": [
                    {
                        "step_index": 2 if decision == "applied" else 1,
                        "behavior": "Synthetic reviewed action",
                        "clause_ids": ["procedure"] if decision == "applied" else [],
                    }
                ],
                "ceremonial_read": False,
                "attribution_limits": ["Synthetic fixture, no causal benefit demonstrated."],
            },
        }
        demos.append(
            {
                "artifact": str(write(directory / "artifact.json", artifact.to_value())),
                "prepared_plan": str(
                    write(
                        directory / "plan.json",
                        {
                            "trajectory_id": name,
                            "tokenizer_id": plan.tokenizer_id,
                            "initial_text": plan.initial_text,
                            "edges": plan.wire_edges(),
                        },
                    )
                ),
                "annotation": str(write(directory / "annotation.json", note)),
                "source": ["synthetic", name],
            }
        )
    return write(
        root / "corpus.json",
        {
            "format": entry.STRUCTURED_CORPUS_FORMAT,
            "candidate": {
                "library": library.to_value(),
                "sampling": {"synthetic-phase": 7},
                "collection_condition": {"kind": "scripted-synthetic-fixture"},
            },
            "demonstrations": demos,
            "development_sources": [
                ["synthetic", x] for x in ("applied", "applied-second", "not-called")
            ],
            "excluded_sources": [["synthetic", "iid-source"]],
            "isolation_ref": "synthetic-frozen-split",
            "task_sources": [
                [x, "synthetic", x] for x in ("applied", "applied-second", "not-called")
            ],
        },
    )


@pytest.mark.parametrize("review_kind", ["human", "model-assisted-public-behavior"])
def test_cli_warmup_resume_export_to_clean_new_ttb_not_original_step0(
    tmp_path, monkeypatch, training_backbone_config, review_kind
):
    config = QwenMultimodalBackboneConfig.from_value(
        replace(training_backbone_config, torch_dtype="bfloat16", lora_dropout=0).to_value()
    )
    causal = QwenBackboneConfig.from_value(config.to_value())

    def builder(_):
        return QwenPolicyBackbone(causal)

    monkeypatch.setattr(prepare, "build_qwen_policy_backbone", builder)
    monkeypatch.setattr(entry, "build_qwen_policy_backbone", builder)
    preparation = prepare.prepare_fresh_restart(
        backbone_config=config, output_root=tmp_path / "base"
    )
    _, binding = _read_preparation(preparation)
    original = builder(config)
    original.load_checkpoint(binding.directory)
    base_fixture = build_application_fixture(
        tmp_path / "library-fixture", causal, backbone=original
    )
    library = base_fixture.application.library.state
    corpus = corpus_files(tmp_path / "corpus", original, library, review_kind)
    root = tmp_path / "warmup"
    finish = entry.finish

    def first_update(trainer, root):
        trainer.step()
        checkpoint = root / "update-00000001"
        trainer.save(checkpoint)
        return checkpoint

    monkeypatch.setattr(entry, "finish", first_update)
    checkpoint = entry.run(
        preparation=preparation,
        corpus_file=corpus,
        output_root=root,
        config=WarmupConfig(0.01, 2),
        run_id="synthetic",
    )
    saved_run = json.loads((root / "run.json").read_text())
    for source, archived in saved_run["input_relocations"].items():
        from pathlib import Path

        assert (root / archived).read_bytes() == Path(source).read_bytes()
    # Recovery consumes saved reviews and tokens, not mutable original inputs.
    for note in saved_run["application_review"]["annotations"]:
        Path(note["source"]).unlink()
    monkeypatch.setattr(entry, "finish", finish)
    original_bytes = checkpoint.joinpath("warmup.json").read_bytes()
    final = entry.resume(root=root, checkpoint=checkpoint)
    assert final.name == "update-00000002"
    assert checkpoint.joinpath("warmup.json").read_bytes() == original_bytes
    with pytest.raises(ValueError):
        entry.resume(root=root, checkpoint=checkpoint)
    output = entry.export(root=root, checkpoint=final, output_root=tmp_path / "ttb-initial")
    declaration = initialization_condition(output)["initialization"]
    assert declaration["original_step_zero"] is False
    assert declaration["warmup_updates"] == 2
    assert (
        declaration["candidate"]["application_review"][0]["original"]["review_kind"] == review_kind
    )
    require_initialization_candidate(output, sampling={"synthetic-phase": 7})
    with pytest.raises(ValueError):
        require_initialization_candidate(output, sampling={"synthetic-phase": 99})
    parsed, warmed_binding = _read_preparation(output)
    assert parsed == config
    assert json.loads(output.read_text())["format"] == WARMUP_PREPARATION_FORMAT
    assert not (output.parent / "optimizer.pt").exists()
    initial = builder(config)
    initial.load_checkpoint(warmed_binding.directory)
    initial.bind_initial_trainable_state(warmed_binding.trainable_state)
    fixture = build_application_fixture(tmp_path / "new-ttb", causal, backbone=initial)
    documents = tuple(library.documents.values())
    monkeypatch.setattr(
        _evolution_preflight_seed, "planned_seed_documents", lambda profile: documents
    )
    require_clean_initial_application(
        fixture.application,
        preparation=output,
        checkpoint_directory=output.parent / "initial-policy",
        root=tmp_path / "new-ttb",
    )
    report = json.loads((tmp_path / "new-ttb" / "fresh-application-start.json").read_text())
    assert report["format"] == "warmup-initialized-application-start@1"
    assert report["checks"]["forward_default_lora_b_zero"] is False
    assert report["checks"]["optimizer_state_empty"] is True
    assert report["checks"]["posterior_empty"] is True
    assert report["checks"]["detector_fresh"] is True
    assert report["checks"]["task_cursor_zero"] is True
    assert report["checks"]["backward_z_match_original_preparation"] is True
    from skillev.training.fresh_state import require_fresh_state

    with pytest.raises(ValueError):
        require_fresh_state(report)
    # The alternate initialization branch does not forgive an actual stale Adam.
    parameter = next(iter(fixture.application.training_loop.optimizer.param_groups[0]["params"]))
    fixture.application.training_loop.optimizer.state[parameter]["step"] = torch.tensor(1.0)
    rejected_root = tmp_path / "rejected-ttb"
    rejected_root.mkdir()
    with pytest.raises(ValueError):
        require_clean_initial_application(
            fixture.application,
            preparation=output,
            checkpoint_directory=output.parent / "initial-policy",
            root=rejected_root,
        )


def test_no_corpus_or_unattributed_review_never_constructs_model(tmp_path, monkeypatch):
    monkeypatch.setattr(
        entry, "build_qwen_policy_backbone", lambda _: pytest.fail("must not build")
    )
    corpus = write(tmp_path / "empty.json", {"format": entry.CORPUS_FORMAT, "demonstrations": []})
    with pytest.raises(ValueError):
        entry.run(
            preparation=tmp_path / "absent",
            corpus_file=corpus,
            output_root=tmp_path / "out",
            config=WarmupConfig(0.01, 1),
            run_id="missing",
        )
    annotation = write(tmp_path / "annotation.json", {"review_kind": "invented-approval"})
    write(
        corpus, {"format": entry.CORPUS_FORMAT, "demonstrations": [{"annotation": str(annotation)}]}
    )
    with pytest.raises(ValueError):
        entry.run(
            preparation=tmp_path / "absent",
            corpus_file=corpus,
            output_root=tmp_path / "out",
            config=WarmupConfig(0.01, 1),
            run_id="missing",
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("command", ["run", "resume", "export"])
def test_cli_dispatches_explicit_paths_and_optimizer_controls(tmp_path, monkeypatch, command):
    received = {}

    def operation(**kwargs):
        received.update(kwargs)
        return tmp_path / "result"

    monkeypatch.setattr(entry, command, operation)
    if command == "run":
        arguments = [
            "run",
            "--preparation",
            str(tmp_path / "prep"),
            "--corpus",
            str(tmp_path / "corpus"),
            "--output-root",
            str(tmp_path / "output"),
            "--run-id",
            "declared",
            "--learning-rate",
            "0.001",
            "--updates",
            "2",
        ]
    else:
        arguments = [
            command,
            "--run-root",
            str(tmp_path / "run"),
            "--checkpoint",
            str(tmp_path / "run/update-00000002"),
        ]
        if command == "export":
            arguments += ["--output-root", str(tmp_path / "ttb")]
    assert entry.main(arguments) == 0
    if command == "run":
        assert received["config"] == WarmupConfig(0.001, 2)
        assert received["run_id"] == "declared"
    else:
        assert received["checkpoint"] == tmp_path / "run/update-00000002"
