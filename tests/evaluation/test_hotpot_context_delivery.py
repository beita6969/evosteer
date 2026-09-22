"""All released passages survive source projection and actual owner requests."""

import asyncio
import json

import pytest
from skillev_private.evaluation.integrity_sources import public_source_view

from skillev.evaluation.input_metric_contracts import PublicTaskView
from skillev.evaluation.public_context import ContextCapacityIncompleteError
from skillev.evaluation.sealed_candidates import EventOrigin
from skillev.evaluation.step0_integrity import InferenceArm
from tests.evaluation.test_integrity_broker_boundary import ServingFixture, runtime


def source_row(paired):
    passages = [
        [f"Public title {index}", [f"First public sentence {index}.", "Second sentence: café."]]
        if paired
        else f"[Public title {index}] First public sentence {index}. Second sentence: café."
        for index in range(10)
    ]
    return {
        "question": "Source wrapper.\nQuestion: Which synthetic public title is requested?",
        "context": passages,
        "answer": "SCORER_ONLY_REFERENCE",
        "supporting_facts": [["SCORER_ONLY_SUPPORT", 0]],
    }


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("repair", [False, True])
def test_every_passage_reaches_owner_and_interface_repair(tmp_path, paired, repair):
    source = source_row(paired)
    entry = public_source_view("synthetic-hotpot", "hotpotqa", source)
    responses = ["Final answer: Public title 4"]
    if repair:
        responses.insert(0, "Final answer: Public title 2\nFinal answer: Public title 3")
    instance = runtime(tmp_path, entry, responses)

    class RecordingServing(ServingFixture):
        def __init__(self):
            super().__init__(responses)
            self.requests = []

        async def generate_evaluation(self, request, **kwargs):
            self.requests.append(request)
            return await super().generate_evaluation(request, **kwargs)

    serving = RecordingServing()
    instance.generators = [serving]
    scope = ("synthetic", "A2", entry.task_id)
    try:
        final = asyncio.run(instance.generate(entry, InferenceArm("A2"), "synthetic"))
        assert final.text == "Public title 4"
        assert len(serving.requests) == 1 + int(repair)
        assert final.intervention_counts["peer_model_calls"] == 0
        assert json.loads(dict(entry.fields)["context"]) == source["context"]
        traces = instance.journal.traces(
            scope, "rendered-request", origin=EventOrigin.ACTOR_DIAGNOSTIC
        )
        for trace, request in zip(traces, serving.requests, strict=True):
            messages = tuple(trace["messages"])
            text = "\n".join(row["content"] for row in messages)
            instruction = "\n".join(row["content"] for row in messages if row["role"] == "system")
            # Public answer semantics survive both the first call and repair;
            # no score threshold or reference-specific answer hint is supplied.
            assert "answer span" in instruction
            assert "yes/no" in instruction
            assert "F1" not in instruction
            assert entry.render() in text
            assert "Which synthetic public title is requested?" in text
            for passage in source["context"]:
                pieces = [passage[0], *passage[1]] if paired else [passage]
                assert all(piece in text for piece in pieces)
            assert "SCORER_ONLY_REFERENCE" not in text
            assert "SCORER_ONLY_SUPPORT" not in text
            assert (
                tuple(
                    instance.tokenizer.encode_integrity_messages(
                        messages, enable_thinking=False, tools=tuple(trace["tools"])
                    )
                )
                == request.input_ids
            )
            assert trace["archived_message_count"] == 0
        instance.validate_candidate(instance.journal, entry, InferenceArm("A2"), "synthetic")
    finally:
        instance.journal.close()


def test_insufficient_capacity_does_not_silently_fall_back_to_question_only(tmp_path):
    entry = public_source_view("synthetic-hotpot", "hotpotqa", source_row(False))
    instance = runtime(tmp_path, entry, ["Final answer: Public title 4"])
    instance.config["context_length"] = 600
    try:
        with pytest.raises((ContextCapacityIncompleteError, RuntimeError)):
            asyncio.run(instance.generate(entry, InferenceArm("A2"), "synthetic"))
        assert not instance.generators[0].profiles
    finally:
        instance.journal.close()


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("benchmark", ["hotpotqa", "triviaqa"])
def test_deliberation_guidance_only_reaches_thinking_hotpot_owner(tmp_path, benchmark, thinking):
    entry = (
        public_source_view("synthetic-hotpot", benchmark, source_row(False))
        if benchmark == "hotpotqa"
        else PublicTaskView.from_record(
            "synthetic-trivia",
            benchmark,
            {"question": "Which public title?", "public_context": "Public title 4 is requested."},
        )
    )
    answer = "Final answer: Public title 4"
    response = f"The public passage identifies the title.</think>{answer}" if thinking else answer
    instance = runtime(tmp_path, entry, [response])
    arm = InferenceArm("A2", native_thinking=thinking)
    scope = ("synthetic", "A2", entry.task_id)
    try:
        final = asyncio.run(instance.generate(entry, arm, "synthetic"))
        assert final.text == "Public title 4"
        assert final.intervention_counts["model_calls"] == 1
        assert final.intervention_counts["peer_model_calls"] == 0
        traces = instance.journal.traces(
            scope, "rendered-request", origin=EventOrigin.ACTOR_DIAGNOSTIC
        )
        instruction = "\n".join(
            row["content"] for row in traces[0]["messages"] if row["role"] == "system"
        )
        assert ("reason carefully" in instruction) == (thinking and benchmark == "hotpotqa")
        assert "SCORER_ONLY_REFERENCE" not in instruction
        assert "SCORER_ONLY_SUPPORT" not in instruction
        instance.validate_candidate(instance.journal, entry, arm, "synthetic")
    finally:
        instance.journal.close()
