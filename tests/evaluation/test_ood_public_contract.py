"""OOD source identity, shared semantics and source-to-token public delivery."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from skillev_private.evaluation.ood_export import project_row

from skillev.evaluation.input_identity import SourceIdentity
from skillev.evaluation.input_metric_contracts import CONTRACTS, OOD_BENCHMARKS, PublicTaskView
from skillev.evaluation.integrity_actor import shared_task_instruction
from skillev.evaluation.public_input_receipt import describe_public_input
from skillev.evaluation.step0_integrity import InferenceArm
from skillev.task_semantic_guidance import PUBLIC_TASK_SEMANTICS_V5, TRAINING_PUBLIC_INPUT


def test_input_receipt_accepts_boundary_trim_but_not_missing_option_content():
    entry = PublicTaskView.from_record(
        "synthetic",
        "gpqa-diamond-bioorganic",
        {"question": "Question?", "options": "A. First\nB. Second\nC. Third\nD. Fourth "},
    )
    complete = describe_public_input(
        entry,
        entry.render().strip(),
        input_tokens=20,
        omitted_history_messages=0,
    )
    missing = describe_public_input(
        entry,
        entry.render().replace("C. Third\n", "").strip(),
        input_tokens=18,
        omitted_history_messages=0,
    )
    assert complete.required_input_received
    assert not missing.required_input_received


@pytest.mark.parametrize("benchmark", OOD_BENCHMARKS)
def test_shared_ood_meaning_does_not_depend_on_training_or_evaluation_entry(benchmark):
    spec = CONTRACTS[benchmark]
    native = spec.task_semantics("released-ood-source@1", version=PUBLIC_TASK_SEMANTICS_V5)
    training = spec.task_semantics(TRAINING_PUBLIC_INPUT, version=PUBLIC_TASK_SEMANTICS_V5)
    assert native == training
    view = PublicTaskView.from_training_task(
        SimpleNamespace(
            task_id="synthetic",
            query="Public task",
            public_context={"benchmark_id": benchmark},
            source_messages=(),
        )
    )
    arm = InferenceArm("shared-ood", task_semantic_guidance=PUBLIC_TASK_SEMANTICS_V5)
    assert native in shared_task_instruction(view, arm)
    assert spec.budget_semantics
    assert spec.termination
    assert spec.parser
    assert spec.metric
    with pytest.raises(ValueError):
        spec.task_semantics("released-iid-source@1", version=PUBLIC_TASK_SEMANTICS_V5)


def test_ood_identity_is_not_an_iid_or_an_exposure_label():
    view = PublicTaskView.from_record("id", "nq-open", {"question": "A public question"})
    identity = SourceIdentity.for_entry(
        view,
        {"nq-open": {"source": "public-dataset", "split": "dev"}},
        panel_id="frozen-panel",
        exposure_status="inspected-development",
    )
    assert identity.panel_id == "frozen-panel"
    assert identity.population_id == "public-dataset"
    assert identity.source_revision is None
    assert identity.source_split == "dev"
    assert identity.exposure_status == "inspected-development"
    with pytest.raises(ValueError):
        replace(view, input_profile="released-iid-source@1")


def test_musique_duplicate_titles_preserve_every_paragraph_and_no_private_labels():
    row = {
        "question": "Which fictional tree?",
        "answer": "private reference",
        "answer_aliases": [],
        "question_decomposition": [{"answer": "private decomposition"}],
        "paragraphs": [
            {
                "idx": 3,
                "title": "Same title",
                "paragraph_text": "First public paragraph.",
                "is_supporting": True,
            },
            {
                "idx": 8,
                "title": "Same title",
                "paragraph_text": "Second public paragraph.",
                "is_supporting": False,
            },
        ],
    }
    projected = project_row("musique", 0, row)
    view = PublicTaskView.from_record(projected["task_id"], "musique", projected["public"])
    assert "private" not in view.render()
    assert "is_supporting" not in view.render()
    paragraphs = projected["public_input_receipt"]["paragraphs"]
    receipt = describe_public_input(
        view,
        view.render(),
        input_tokens=100,
        omitted_history_messages=0,
        source_paragraphs=paragraphs,
    )
    assert receipt.source_paragraph_count == 2
    assert receipt.paragraph_ids == ("3", "8")
    assert receipt.exported_source_complete
    assert receipt.required_input_received
    assert receipt.paragraph_order_received
    missing = describe_public_input(
        view,
        "First public paragraph.",
        input_tokens=5,
        omitted_history_messages=1,
        source_paragraphs=paragraphs,
    )
    assert not missing.required_input_received
    assert not missing.paragraph_order_received


def test_legacy_source_without_paragraph_metadata_is_not_claimed_source_verified():
    view = PublicTaskView.from_record("id", "musique", {"question": "Q", "context": "P"})
    receipt = describe_public_input(
        view, view.render(), input_tokens=10, omitted_history_messages=0
    )
    assert receipt.required_input_received
    assert receipt.source_paragraph_count is None
    assert receipt.exported_source_complete is None


def test_shared_receipt_does_not_drop_native_trivia_reading_context():
    view = PublicTaskView.from_record(
        "id", "triviaqa", {"question": "Question?", "public_context": "Public evidence."}
    )
    complete = describe_public_input(
        view, view.render(), input_tokens=10, omitted_history_messages=0
    )
    question_only = describe_public_input(
        view, "Question?", input_tokens=3, omitted_history_messages=0
    )
    assert complete.required_input_received
    assert not question_only.required_input_received
