"""Separate NQ corpus condition; real isolated owner, literal queries and fixed EM."""

import asyncio
import csv
import gzip
import json
from dataclasses import asdict, replace

import pytest
from skillev_private.evaluation.ood_retrieval import CorpusSearchSession
from skillev_private.evaluation.wikipedia_corpus import build_index, corpus_metadata, search_corpus

from skillev.evaluation.corpus_search import (
    CORPUS_ID,
    GLASGOW_QUERY_POLICY,
    INPUT_PROFILE,
    PHRASE_QUERY_POLICY,
    REQUIRED_PHRASE_QUERY_POLICY,
    SNAPSHOT_DATE,
    SOFT_CONTEXT_QUERY_POLICY,
    CorpusSearchProfile,
)
from skillev.evaluation.input_metric_contracts import PublicTaskView
from skillev.evaluation.native_tool_calls import native_control_payload
from skillev.evaluation.sealed_candidates import CandidateJournal
from skillev.evaluation.step0_integrity import InferenceArm, ToolCallMode
from skillev.task_semantic_guidance import (
    PUBLIC_TASK_SEMANTICS_V8,
    PUBLIC_TASK_SEMANTICS_V9,
    public_task_semantics,
)
from tests.evaluation.test_integrity_broker_boundary import runtime
from tests.evaluation.test_integrity_native_tool_calls import call


def corpus(tmp_path, corpus_id="synthetic-public-corpus"):
    source = tmp_path / "public.tsv.gz"
    with gzip.open(source, "wt", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("id", "text", "title"))
        writer.writerow((1, "The fictional copper city is named Elmport.", "Copper city"))
        writer.writerow((2, "A fictional plant has blue leaves.", "Blue leaves"))
    path = tmp_path / "corpus.sqlite"
    assert build_index(source, path, corpus_id=corpus_id) == 2
    return {
        "index_path": str(path),
        "public_profile": asdict(CorpusSearchProfile(corpus_id=corpus_id, maximum_queries=1)),
    }


def test_dpr_snapshot_metadata_reaches_owner_without_inventing_question_date(tmp_path):
    from pathlib import Path

    settings = corpus(tmp_path, corpus_id=CORPUS_ID)
    profile = CorpusSearchProfile(**settings["public_profile"])
    metadata = corpus_metadata(Path(settings["index_path"]))
    assert metadata["snapshot_date"] == SNAPSHOT_DATE
    assert SNAPSHOT_DATE in profile.instruction()
    assert "not the date of an individual question" in profile.instruction()
    assert SNAPSHOT_DATE not in CorpusSearchProfile(corpus_id="custom-corpus").instruction()
    journal = CandidateJournal(tmp_path / "journal.sqlite")
    scope = ("run", "arm", "nq")
    CorpusSearchSession(settings, journal, scope)
    recorded = journal.traces(scope, "corpus-profile")[0]
    assert recorded["index_metadata"]["snapshot_date"] == SNAPSHOT_DATE
    journal.close()


def test_previous_wrong_snapshot_metadata_cannot_be_relabelled_silently(tmp_path):
    import sqlite3
    from pathlib import Path

    settings = corpus(tmp_path, corpus_id=CORPUS_ID)
    path = Path(settings["index_path"])
    path.chmod(0o600)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE metadata SET value=? WHERE key='snapshot_date'", ("2020-03-17",))
    journal = CandidateJournal(tmp_path / "journal.sqlite")
    with pytest.raises(ValueError):
        CorpusSearchSession(settings, journal, ("run", "arm", "nq"))
    journal.close()


def test_full_corpus_import_is_question_independent_and_read_only(tmp_path):
    settings = corpus(tmp_path)
    from pathlib import Path

    path = Path(settings["index_path"])
    assert corpus_metadata(path)["passages"] == "2"
    rows = search_corpus(path, "which copper city", limit=1)
    assert rows[0]["passage_id"] == "1"
    assert "Elmport" in rows[0]["text"]
    assert not search_corpus(path, "unrelatedword", limit=5)
    assert not search_corpus(path, "which is the", limit=5)
    # Standard English function words no longer expand OR into most of Wikipedia.
    assert not search_corpus(path, "I me it's myself", limit=5)
    assert search_corpus(path, "I am asking about the copper city", limit=1)[0]["passage_id"] == "1"
    with pytest.raises(FileExistsError):
        build_index(tmp_path / "public.tsv.gz", path)


@pytest.mark.parametrize("query_policy", [PHRASE_QUERY_POLICY, GLASGOW_QUERY_POLICY])
def test_quoted_phrase_keeps_common_words_without_changing_legacy_search(tmp_path, query_policy):
    source = tmp_path / "phrases.tsv.gz"
    with gzip.open(source, "wt", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("id", "text", "title"))
        writer.writerow((1, "One fictional story set in Elmport.", "To Me And You"))
        writer.writerow((2, "An unrelated fictional story.", "Story"))
        writer.writerow((3, "A different fictional story.", "You And Me"))
    path = tmp_path / "phrases.sqlite"
    build_index(source, path, corpus_id="synthetic-phrases")
    query = '"to me and you"'
    assert not search_corpus(path, query, limit=5)
    options = {"limit": 5, "query_policy": query_policy}
    assert [r["passage_id"] for r in search_corpus(path, query, **options)] == ["1"]
    assert search_corpus(path, query + " story", **options)[0]["passage_id"] == "1"
    assert not search_corpus(path, '"me you and to"', **options)
    assert not search_corpus(path, "to me and you", **options)
    if query_policy == GLASGOW_QUERY_POLICY:
        assert not search_corpus(path, "one", **options)
        assert search_corpus(path, '"one"', **options)
    with pytest.raises(ValueError):
        search_corpus(path, query, limit=5, query_policy="unknown")


@pytest.mark.parametrize("semantics", [PUBLIC_TASK_SEMANTICS_V8, PUBLIC_TASK_SEMANTICS_V9])
@pytest.mark.parametrize(
    "query_policy",
    [PHRASE_QUERY_POLICY, REQUIRED_PHRASE_QUERY_POLICY, SOFT_CONTEXT_QUERY_POLICY],
)
def test_phrase_search_flows_through_real_isolated_owner_and_budget(
    tmp_path, semantics, query_policy
):
    settings = corpus(tmp_path)
    settings["public_profile"]["query_policy"] = query_policy
    entry = replace(
        PublicTaskView.from_record("nq", "nq-open", {"question": "Name the fictional place."}),
        input_profile=INPUT_PROFILE,
    )
    query = '"copper city"'
    instance = runtime(
        tmp_path, entry, [call("corpus_search", query=query), "Final answer: Elmport"]
    )
    instance.config["corpus_retrieval"] = {"nq-open": settings}
    arm = InferenceArm(
        "phrase",
        tool_call_mode=ToolCallMode.QWEN_XML,
        task_semantic_guidance=semantics,
    )
    final = asyncio.run(instance.generate(entry, arm, "quoted"))
    assert final.text == "Elmport"
    assert final.intervention_counts["tool_calls"] == 1
    assert not final.intervention_counts["peer_model_calls"]
    scope = ("quoted", "phrase", "nq")
    event = instance.journal.traces(scope, "corpus-search-result")[0]
    assert event["query"] == query
    assert event["result"]["passages"][0]["passage_id"] == "1"
    requests = instance.journal.traces(scope, "rendered-request")
    assert (
        CorpusSearchProfile(**settings["public_profile"]).instruction()
        in (requests[0]["messages"][0]["content"])
    )
    assert (
        public_task_semantics("nq-open", input_profile=INPUT_PROFILE, version=semantics)
        in requests[0]["messages"][0]["content"]
    )
    assert any(
        json.dumps(event["result"], ensure_ascii=False) in item["content"]
        for item in requests[1]["messages"]
    )
    instance.journal.close()


def test_profile_changes_semantics_without_changing_question_projection():
    closed = PublicTaskView.from_record("nq", "nq-open", {"question": "Synthetic question?"})
    retrieval = replace(closed, input_profile=INPUT_PROFILE)
    assert retrieval.render() == closed.render()
    kwargs = {"version": PUBLIC_TASK_SEMANTICS_V8}
    assert "closed-book" in public_task_semantics(
        "nq-open", input_profile=closed.input_profile, **kwargs
    )
    assert "corpus search" in public_task_semantics(
        "nq-open", input_profile=INPUT_PROFILE, **kwargs
    )
    with pytest.raises(ValueError):
        replace(
            PublicTaskView.from_record("math", "math-hard", {"problem": "x?"}),
            input_profile=INPUT_PROFILE,
        )


def test_required_quotes_do_not_expand_a_single_common_word_into_all_passages(tmp_path):
    source = tmp_path / "required.tsv.gz"
    with gzip.open(source, "wt", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("id", "text", "title"))
        writer.writerow((1, "A is a fictional character in Lantern Harbor.", "Lantern Harbor"))
        writer.writerow((2, "A fictional plant grows beside a lake.", "Other story"))
        writer.writerow((3, "Lantern Harbor features fictional people.", "Lantern Harbor"))
    path = tmp_path / "required.sqlite"
    build_index(source, path, corpus_id="synthetic-required")
    query = '"A" Lantern Harbor character'
    old = search_corpus(path, query, limit=20, query_policy=GLASGOW_QUERY_POLICY)
    assert len(old) == 3
    new = search_corpus(path, query, limit=20, query_policy=REQUIRED_PHRASE_QUERY_POLICY)
    assert [r["passage_id"] for r in new] == ["1"]
    # Both required phrases and their common words survive; no guessed synonym.
    assert (
        search_corpus(
            path, '"A" "Lantern Harbor"', limit=20, query_policy=REQUIRED_PHRASE_QUERY_POLICY
        )[0]["passage_id"]
        == "1"
    )
    assert not search_corpus(
        path, '"A" "Harbor Lantern"', limit=20, query_policy=REQUIRED_PHRASE_QUERY_POLICY
    )
    profile = CorpusSearchProfile(query_policy=REQUIRED_PHRASE_QUERY_POLICY)
    assert "required matches" in profile.instruction()


def test_soft_context_ranks_without_excluding_a_quoted_title(tmp_path):
    source = tmp_path / "soft-context.tsv.gz"
    with gzip.open(source, "wt", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("id", "text", "title"))
        writer.writerow((1, "A fictional band wrote the song.", "Copper River"))
        writer.writerow((2, "A fictional singer wrote the song.", "Copper River"))
        writer.writerow((3, "A fictional singer wrote the song.", "Other Song"))
        writer.writerow((4, "A fictional band wrote the song.", "River Copper"))
    path = tmp_path / "soft-context.sqlite"
    build_index(source, path, corpus_id="synthetic-soft-context")
    query = '"Copper River" singer'
    old = search_corpus(path, query, limit=20, query_policy=REQUIRED_PHRASE_QUERY_POLICY)
    assert [row["passage_id"] for row in old] == ["2"]
    rows = search_corpus(path, query, limit=20, query_policy=SOFT_CONTEXT_QUERY_POLICY)
    assert [row["passage_id"] for row in rows] == ["2", "1"]
    assert rows[1]["text"] == "A fictional band wrote the song."
    # Multiple quotes remain mandatory; they do not admit either phrase alone.
    rows = search_corpus(
        path, '"Copper River" "band" singer', limit=20, query_policy=SOFT_CONTEXT_QUERY_POLICY
    )
    assert [row["passage_id"] for row in rows] == ["1"]
    # A quoted common singleton still intersects its context rather than
    # expanding the search to the whole corpus. Unquoted OR is unchanged.
    rows = search_corpus(path, '"A" singer', limit=20, query_policy=SOFT_CONTEXT_QUERY_POLICY)
    assert {row["passage_id"] for row in rows} == {"2", "3"}
    for query in ('"A" singer', "band singer"):
        assert search_corpus(
            path, query, limit=20, query_policy=SOFT_CONTEXT_QUERY_POLICY
        ) == search_corpus(path, query, limit=20, query_policy=REQUIRED_PHRASE_QUERY_POLICY)


def test_real_actor_uses_one_literal_search_then_its_own_final(tmp_path):
    settings = corpus(tmp_path)
    entry = replace(
        PublicTaskView.from_record(
            "nq", "nq-open", {"question": "What is the copper city called?"}
        ),
        input_profile=INPUT_PROFILE,
    )
    response = call("corpus_search", query="copper city")
    instance = runtime(tmp_path, entry, [response, "Final answer: Elmport"])
    instance.config["corpus_retrieval"] = {"nq-open": settings}
    arm = InferenceArm(
        "A2", tool_call_mode=ToolCallMode.QWEN_XML, task_semantic_guidance=PUBLIC_TASK_SEMANTICS_V8
    )
    final = asyncio.run(instance.generate(entry, arm, "synthetic"))
    assert final.text == "Elmport"
    assert final.intervention_counts["tool_calls"] == 1
    assert final.intervention_counts["model_calls"] == 2
    assert not final.intervention_counts["peer_model_calls"]
    scope = ("synthetic", "A2", "nq")
    events = instance.journal.traces(scope, "corpus-search-result")
    assert events[0]["query"] == "copper city"
    delivered = json.dumps(events[0]["result"], ensure_ascii=False)
    requests = instance.journal.traces(scope, "rendered-request")
    assert any(delivered in item["content"] for item in requests[1]["messages"])
    assert "index_path" not in json.dumps(requests)
    instance.journal.close()


def test_closed_book_cannot_silently_activate_corpus(tmp_path):
    entry = PublicTaskView.from_record("nq", "nq-open", {"question": "Synthetic question?"})
    instance = runtime(
        tmp_path, entry, [call("corpus_search", query="anything"), "Final answer: Elmport"]
    )
    final = asyncio.run(instance.generate(entry, InferenceArm("A2"), "closed"))
    assert final.text == "Elmport"
    assert not instance.journal.traces(("closed", "A2", "nq"), "corpus-search-start")
    instance.journal.close()


def test_owner_query_batch_is_delivered_in_order_and_charged_per_query(tmp_path):
    settings = corpus(tmp_path)
    settings["public_profile"]["maximum_queries"] = 2
    entry = replace(
        PublicTaskView.from_record("nq", "nq-open", {"question": "Describe two places."}),
        input_profile=INPUT_PROFILE,
    )
    response = (
        call("corpus_search", query="copper city")
        + "\n"
        + call("corpus_search", query="blue leaves")
    )
    instance = runtime(tmp_path, entry, [response, "Final answer: Elmport"])
    instance.config["corpus_retrieval"] = {"nq-open": settings}
    arm = InferenceArm(
        "A2", tool_call_mode=ToolCallMode.QWEN_XML, task_semantic_guidance=PUBLIC_TASK_SEMANTICS_V8
    )
    final = asyncio.run(instance.generate(entry, arm, "batch"))
    assert final.text == "Elmport"
    assert final.intervention_counts["tool_calls"] == 2
    assert final.intervention_counts["model_calls"] == 2
    scope = ("batch", "A2", "nq")
    events = instance.journal.traces(scope, "corpus-search-result")
    assert [e["query"] for e in events] == ["copper city", "blue leaves"]
    assert [e["result"]["remaining_queries"] for e in events] == [1, 0]
    messages = instance.journal.traces(scope, "rendered-request")[1]["messages"]
    for event in events:
        assert any(
            json.dumps(event["result"], ensure_ascii=False) in m["content"] for m in messages
        )
    instance.journal.close()


def test_query_batches_do_not_select_among_mixed_actions_or_final_answers():
    search = call("corpus_search", query="copper city")
    for other in (call("act", command="look around"), call("submit_answer", answer="Elmport")):
        with pytest.raises(ValueError):
            native_control_payload(search + "\n" + other)
    with pytest.raises(ValueError):
        native_control_payload("Final answer: Elmport\n" + search)


def test_batch_cannot_reorder_queries_or_reset_the_episode_search_budget(tmp_path):
    settings = corpus(tmp_path)
    journal = CandidateJournal(tmp_path / "journal.sqlite")
    session = CorpusSearchSession(settings, journal, ("run", "arm", "nq"))
    response = (
        call("corpus_search", query="copper city")
        + "\n"
        + call("corpus_search", query="blue leaves")
    )

    async def execute():
        kwargs = {
            "response": response,
            "call_id": "batch",
            "participant": "owner",
            "final_ready": False,
        }
        with pytest.raises(ValueError):
            await session.execute("blue leaves", **kwargs)
        assert (await session.execute("copper city", **kwargs))["passages"]
        denied = await session.execute("blue leaves", **kwargs)
        assert "error" in denied
        assert not denied["remaining_queries"]
        with pytest.raises(ValueError):
            await session.execute("copper city", **kwargs)

    asyncio.run(execute())
    journal.close()


def test_budget_is_enforced_and_broker_does_not_rewrite_queries(tmp_path):
    settings = corpus(tmp_path)
    journal = CandidateJournal(tmp_path / "journal.sqlite")
    session = CorpusSearchSession(settings, journal, ("run", "arm", "nq"))
    response = call("corpus_search", query="copper city")

    async def execute():
        with pytest.raises(ValueError):
            await session.execute(
                "different query",
                response=response,
                call_id="call1",
                participant="owner",
                final_ready=False,
            )
        first = await session.execute(
            "copper city",
            response=response,
            call_id="call1",
            participant="owner",
            final_ready=False,
        )
        assert first["passages"]
        with pytest.raises(ValueError):
            await session.execute(
                "copper city",
                response=response,
                call_id="call1",
                participant="owner",
                final_ready=False,
            )
        denied = await session.execute(
            "copper city",
            response=response,
            call_id="call2",
            participant="owner",
            final_ready=False,
        )
        assert "error" in denied
        assert not denied["remaining_queries"]

    asyncio.run(execute())
    journal.close()
