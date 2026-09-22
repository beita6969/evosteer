"""Synthetic public-behavior review contracts; no actual model efficacy evidence."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from skillev.training.skill_use_warmup import SkillUseWarmup, WarmupConfig
from skillev.training.warmup_supervision import validate_application_annotation
from tests.training.test_skill_use_warmup import Model, corpus, example


def edit_note(demo, edit):
    path = Path(demo.annotation_ref)
    note = json.loads(path.read_text())
    edit(note)
    path.write_text(json.dumps(note))
    return note


def test_original_token_partition_does_not_turn_visibility_into_application(tmp_path):
    data = corpus(tmp_path, negative="abandoned-after-read")
    original = data.to_value()
    report = data.supervision_report()
    counts = report["categories"]
    assert counts["actual_read_skill"]["actions"] == 3
    assert counts["reviewed_body_visible_application"]["actions"] == 2
    assert counts["other_pre_read_or_unconfirmed"]["actions"] == 1
    assert report["trajectories"][-1]["body_visible_following_actions"] == 1
    assert sum(v["action_token_fraction"] for v in counts.values()) == pytest.approx(1)
    assert report["original_action_tokens"] == sum(
        len(step.action_token_ids) for d in data.demonstrations for step in d.artifact.record.steps
    )
    assert data.to_value() == original
    assert report["multiple_independent_positives"]
    assert not report["weights_applied"]


def test_no_read_direct_and_legacy_missing_reviews_remain_diagnostic(tmp_path):
    data = corpus(tmp_path)
    edit_note(
        data.demonstrations[0], lambda note: note.update(format="skill-application-annotation@1")
    )
    Path(data.demonstrations[1].annotation_ref).unlink()
    report = data.supervision_report()
    assert report["categories"]["reviewed_no_read_direct"]["actions"] == 1
    assert report["categories"]["reviewed_body_visible_application"]["actions"] == 0
    assert report["categories"]["other_pre_read_or_unconfirmed"]["actions"] == 2
    assert not report["complete_structured_reviews"]
    assert report["unreviewed_application_count"] is None
    assert report["trajectories"][0]["ceremonial_read"] is None
    with pytest.raises((ValueError, OSError)):
        SkillUseWarmup(backbone=Model(), corpus=data, config=WarmupConfig(0.01, 1), run_id="new")


@pytest.mark.parametrize("defect", ["quote", "step", "ceremonial", "empty", "missing"])
def test_new_positive_requires_attributable_actual_method_behavior(tmp_path, defect):
    data = corpus(tmp_path)

    def corrupt(note):
        evidence = note["behavior_evidence"]
        if defect == "quote":
            evidence["method_clauses"][0]["quote"] = "NOT_IN_ACTUAL_RETURN"
        elif defect == "step":
            evidence["actions"][0]["step_index"] = 1
        elif defect == "ceremonial":
            evidence["ceremonial_read"] = True
        elif defect == "empty":
            evidence["attribution_limits"] = []
        else:
            note.pop("behavior_evidence")

    edit_note(data.demonstrations[0], corrupt)
    model = Model()
    with pytest.raises(ValueError):
        SkillUseWarmup(backbone=model, corpus=data, config=WarmupConfig(0.01, 1), run_id="bad")
    assert not model.calls
    assert model.f.grad is None


def test_shared_review_can_describe_failed_ceremonial_read_and_caller_metadata(tmp_path):
    demo = example(tmp_path, "applied", success=False)
    note = edit_note(demo, lambda v: v.update(position=3, selection_relevance="unrelated"))
    note["behavior_evidence"]["ceremonial_read"] = True
    note["approved_for_warmup"] = False
    assert validate_application_annotation(note, demo.artifact) is note


def test_two_occurrences_of_one_canonical_positive_are_not_independent(tmp_path):
    data = corpus(tmp_path)
    data = replace(data, aliases=(("synthetic", "applied-second", "applied"),))
    with pytest.raises(ValueError):
        data.for_training()


def test_real_reference_required_new_but_saved_contents_support_recovery(tmp_path):
    data = corpus(tmp_path)
    model = Model()
    trainer = SkillUseWarmup(
        backbone=model, corpus=data, config=WarmupConfig(0.01, 2), run_id="saved"
    )
    trainer.step()
    checkpoint = tmp_path / "checkpoint"
    trainer.save(checkpoint)
    for demo in trainer.corpus.demonstrations:
        Path(demo.annotation_ref).unlink()
    with pytest.raises(OSError):
        SkillUseWarmup(
            backbone=Model(), corpus=trainer.corpus, config=WarmupConfig(0.01, 2), run_id="new"
        )
    restored = SkillUseWarmup.restore(
        backbone=Model(),
        corpus=trainer.corpus,
        directory=checkpoint,
        config=WarmupConfig(0.01, 2),
        run_id="saved",
    )
    assert restored.corpus.supervision_report()["complete_structured_reviews"]
    restored.step()


def test_report_cli_reads_legacy_checkpoint_without_model_optimizer_or_admission(
    tmp_path, monkeypatch, training_backbone_config
):
    from skillev_private.experiments import skill_use_warmup as entry

    from tests.rollout.engine_fakes import ByteTokenizer

    data = corpus(tmp_path)
    for demo in data.demonstrations:
        edit_note(demo, lambda note: note.update(format="skill-application-annotation@1"))
    checkpoint = tmp_path / "legacy"
    checkpoint.mkdir()
    (checkpoint / "warmup.json").write_text(json.dumps({"corpus": data.to_value()}))
    original_bytes = (checkpoint / "warmup.json").read_bytes()
    config = tmp_path / "backbone.json"
    config.write_text(json.dumps(training_backbone_config.to_value()))
    monkeypatch.setattr(entry.QwenTokenizerAdapter, "from_config", lambda _: ByteTokenizer())
    monkeypatch.setattr(entry, "_build", lambda _: pytest.fail("report must not build a model"))
    monkeypatch.setattr(entry, "SkillUseWarmup", lambda **_: pytest.fail("no optimizer"))
    report_file = tmp_path / "report.json"
    args = [
        "report",
        "--backbone-config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(report_file),
    ]
    assert entry.main(args) == 0
    report = json.loads(report_file.read_text())
    assert not report["complete_structured_reviews"]
    assert report["trajectories"][0]["body_visible_following_actions"] == 1
    assert report["categories"]["reviewed_body_visible_application"]["actions"] == 0
    assert (checkpoint / "warmup.json").read_bytes() == original_bytes
    with pytest.raises(FileExistsError):
        entry.main(args)


def test_method_clause_quotes_decoded_original_multiline_instruction(tmp_path):
    body = 'SYNTHETIC_BODY\nUse "quoted" public input and \\ separator.'
    demo = example(tmp_path, "applied", body=body)
    note = edit_note(demo, lambda v: v["behavior_evidence"]["method_clauses"][0].update(quote=body))
    assert validate_application_annotation(note, demo.artifact) is note
