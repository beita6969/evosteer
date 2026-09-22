"""Fixed rank fusion is retrieval, not query rewriting or answer selection."""

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from skillev_private.evaluation import hybrid_retrieval, ood_retrieval

from skillev.evaluation.corpus_search import (
    HYBRID_QUERY_POLICY,
    HYBRID_RANKING,
    HYBRID_SEARCH_PROFILE,
    INPUT_PROFILE,
    SOFT_CONTEXT_QUERY_POLICY,
    CorpusSearchProfile,
)
from skillev.evaluation.input_metric_contracts import PublicTaskView
from skillev.evaluation.sealed_candidates import CandidateJournal
from skillev.evaluation.step0_integrity import InferenceArm, ToolCallMode
from skillev.evaluation.trajectory_diagnostics import trajectory_diagnostics
from skillev.task_semantic_guidance import PUBLIC_TASK_SEMANTICS_V9
from tests.evaluation.test_dpr_retrieval import dense_settings
from tests.evaluation.test_integrity_broker_boundary import runtime
from tests.evaluation.test_integrity_native_tool_calls import call


def passage(pid):
    return {"passage_id": pid, "title": f"Title {pid}", "text": f"Public text {pid}."}


def test_hybrid_is_a_separate_declared_condition():
    legacy = CorpusSearchProfile()
    profile = replace(
        legacy,
        profile_id=HYBRID_SEARCH_PROFILE,
        query_policy=HYBRID_QUERY_POLICY,
        passage_ranking=HYBRID_RANKING,
    )
    assert CorpusSearchProfile(**asdict(profile)) == profile
    assert "supervised on NQ training" in profile.instruction()
    assert "BM25" in profile.instruction()
    assert "one query allowance" in profile.instruction()
    assert "hybrid" not in legacy.instruction()
    for field in ("profile_id", "query_policy", "passage_ranking"):
        with pytest.raises(ValueError):
            replace(profile, **{field: getattr(legacy, field)})


def test_rrf_uses_both_ranks_without_changing_source_text():
    a, b, c = [passage(pid) for pid in "abc"]
    fused, diagnostics = hybrid_retrieval.fuse_passages([a, b], [c, b], limit=3)
    assert [row["passage_id"] for row in fused] == ["b", "a", "c"]
    assert [row["rank"] for row in fused] == [1, 2, 3]
    for row, original in zip(fused, [b, a, c], strict=True):
        assert row["text"] == original["text"]
        assert row["title"] == original["title"]
    assert diagnostics["candidates"][0]["rrf_score"] == pytest.approx(2 / 62)
    assert diagnostics["union_candidates"] == 3
    assert "rank" not in b
    assert len(hybrid_retrieval.fuse_passages([a, b], [c, b], limit=1)[0]) == 1
    assert hybrid_retrieval.fuse_passages([a], [], limit=3)[0] == [{**a, "rank": 1}]


def test_hybrid_owner_query_result_budget_and_cost_are_preserved(tmp_path, monkeypatch):
    settings = dense_settings(tmp_path)
    settings["public_profile"].update(
        profile_id=HYBRID_SEARCH_PROFILE,
        query_policy=HYBRID_QUERY_POLICY,
        passage_ranking=HYBRID_RANKING,
    )
    seen = []

    def dense(path, query, **kwargs):
        seen.append(("dense", query))
        assert kwargs["limit"] == 100
        return [passage("1"), passage("2")], {
            "input_tokens": 9,
            "forward_batches": 1,
            "search_seconds": 0.1,
            "model_id": "synthetic-encoder",
        }

    def lexical(path, query, **kwargs):
        seen.append(("lexical", query))
        assert kwargs["limit"] == 100
        assert kwargs["query_policy"] == SOFT_CONTEXT_QUERY_POLICY
        return [passage("2")]

    monkeypatch.setattr(hybrid_retrieval, "search_dense_corpus", dense)
    monkeypatch.setattr(hybrid_retrieval, "search_corpus", lexical)
    entry = replace(
        PublicTaskView.from_record("nq", "nq-open", {"question": "Which fictional city?"}),
        input_profile=INPUT_PROFILE,
    )
    query = 'the "copper city"'
    instance = runtime(tmp_path, entry, [call("corpus_search", query=query), "Final answer: Wrong"])
    instance.config["corpus_retrieval"] = {"nq-open": settings}
    arm = InferenceArm(
        "hybrid",
        tool_call_mode=ToolCallMode.QWEN_XML,
        task_semantic_guidance=PUBLIC_TASK_SEMANTICS_V9,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(ood_retrieval, "_search_pool", lambda: pool)
        final = asyncio.run(instance.generate(entry, arm, "hybrid"))
    assert final.text == "Wrong"
    assert final.intervention_counts["model_calls"] == 2
    assert final.intervention_counts["peer_model_calls"] == 0
    assert seen == [("dense", query), ("lexical", query)]
    scope = ("hybrid", arm.arm_id, entry.task_id)
    result = instance.journal.traces(scope, "corpus-search-result")[0]["result"]
    assert result["remaining_queries"] == 0
    assert result["query"] == query
    assert [row["passage_id"] for row in result["passages"]] == ["2", "1"]
    requests = instance.journal.traces(scope, "rendered-request")
    assert any(
        json.dumps(result, ensure_ascii=False) in m["content"] for m in requests[1]["messages"]
    )
    assert "rrf_score" not in json.dumps(requests)
    assert "encoder_path" not in json.dumps(requests)
    reader = SimpleNamespace(
        get=lambda *args: final,
        model_outputs=lambda scope: (),
        traces=instance.journal.traces,
    )
    costs = trajectory_diagnostics(reader, (scope,), benchmark="nq-open")
    assert costs["retrieval_dense_cost"]["forward_batches"] == 1
    assert costs["retrieval_dense_cost"]["input_tokens"] == 9
    assert costs["retrieval_hybrid_cost"]["queries_by_ranking"] == {HYBRID_RANKING: 1}
    assert costs["retrieval_hybrid_cost"]["union_candidates"] == 2
    instance.journal.close()


@pytest.mark.parametrize("failing_leg", ["dense", "lexical"])
def test_unavailable_leg_does_not_silently_change_the_retriever(tmp_path, monkeypatch, failing_leg):
    def fail(*args, **kwargs):
        raise RuntimeError("search unavailable")

    monkeypatch.setattr(
        hybrid_retrieval,
        "search_dense_corpus",
        fail if failing_leg == "dense" else lambda *args, **kwargs: ([passage("1")], {}),
    )
    monkeypatch.setattr(hybrid_retrieval, "search_corpus", fail)
    with pytest.raises(RuntimeError):
        hybrid_retrieval.search_hybrid_corpus(
            Path(tmp_path), "literal query", limit=5, model_path=tmp_path, index_path=tmp_path
        )


def test_batch_queue_does_not_consume_each_requests_worker_deadline(tmp_path, monkeypatch):
    settings = dense_settings(tmp_path)
    settings["public_profile"].update(
        profile_id=HYBRID_SEARCH_PROFILE,
        query_policy=HYBRID_QUERY_POLICY,
        passage_ranking=HYBRID_RANKING,
    )

    def worker(*args, **kwargs):
        time.sleep(0.03)
        return [passage("1")], {}

    monkeypatch.setattr(hybrid_retrieval, "search_hybrid_corpus", worker)
    monkeypatch.setattr(ood_retrieval, "HYBRID_WORKER_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(ood_retrieval, "SEARCH_QUEUE_TIMEOUT_SECONDS", 2)
    journal = CandidateJournal(tmp_path / "queue.sqlite")
    sessions = [
        ood_retrieval.CorpusSearchSession(settings, journal, ("run", "arm", str(i)))
        for i in range(32)
    ]

    async def run():
        return await asyncio.gather(
            *(session._dense_search("public query", {"call_id": "c1"}) for session in sessions)
        )

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            monkeypatch.setattr(ood_retrieval, "_search_pool", lambda: pool)
            results = asyncio.run(run())
        assert len(results) == 32
        waits = [
            journal.traces(session.scope, "corpus-search-worker-start")[0]["queue_seconds"]
            for session in sessions
        ]
        assert max(waits) > ood_retrieval.HYBRID_WORKER_TIMEOUT_SECONDS
    finally:
        journal.close()
