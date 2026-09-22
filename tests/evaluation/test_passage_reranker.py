"""Frozen relevance only; no second answerer, private targets or hidden retries."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch
from skillev_private.evaluation import ood_retrieval, passage_reranker

from skillev.evaluation.corpus_search import (
    INPUT_PROFILE,
    MINILM_RANKING,
    MINILM_WIDE_RANKING,
    OWNER_QUERY,
    PUBLIC_QUESTION_QUERY,
    RERANK_CANDIDATE_LIMITS,
    CorpusSearchProfile,
)
from skillev.evaluation.input_metric_contracts import PublicTaskView
from skillev.evaluation.sealed_candidates import CandidateJournal
from skillev.evaluation.step0_integrity import InferenceArm, ToolCallMode
from skillev.evaluation.trajectory_diagnostics import trajectory_diagnostics
from skillev.task_semantic_guidance import PUBLIC_TASK_SEMANTICS_V9
from tests.evaluation.test_integrity_broker_boundary import runtime
from tests.evaluation.test_integrity_native_tool_calls import call
from tests.evaluation.test_ood_corpus_search import corpus


def test_ranking_is_stable_and_does_not_extract_or_modify_passages():
    passages = [
        {"passage_id": str(i), "rank": i, "title": f"Title {i}", "text": f"Full text {i}."}
        for i in range(1, 4)
    ]
    ranked = passage_reranker.order_passages(passages, [-2.0, 4.0, 4.0], limit=2)
    assert [p["passage_id"] for p in ranked] == ["2", "3"]
    for rank, row in enumerate(ranked, 1):
        original = passages[int(row["passage_id"]) - 1]
        assert row["text"] == original["text"]
        assert row["title"] == original["title"]
        assert row["bm25_rank"] == original["rank"]
        assert row["rank"] == rank
    assert [p["rank"] for p in passages] == [1, 2, 3]
    assert not passage_reranker.order_passages([], [], limit=20)
    with pytest.raises(ValueError):
        passage_reranker.order_passages(passages, [1.0], limit=2)


def test_reranking_is_an_explicit_condition_and_does_not_change_legacy_defaults():
    legacy = CorpusSearchProfile()
    enhanced = replace(legacy, passage_ranking=MINILM_RANKING)
    assert "reranked" not in legacy.instruction()
    assert "first 100 BM25" in enhanced.instruction()
    assert "does not answer" in enhanced.instruction()
    assert CorpusSearchProfile(**asdict(enhanced)) == enhanced
    with pytest.raises(ValueError):
        replace(legacy, passage_ranking="undeclared-ranker")
    with pytest.raises(ValueError):
        replace(legacy, reranker_query_source=PUBLIC_QUESTION_QUERY)
    question = replace(enhanced, reranker_query_source=PUBLIC_QUESTION_QUERY)
    assert "original public question" in question.instruction()
    assert CorpusSearchProfile(**asdict(question)) == question
    assert CorpusSearchProfile().reranker_query_source == OWNER_QUERY
    wider = replace(enhanced, passage_ranking=MINILM_WIDE_RANKING)
    assert "first 1000 BM25" in wider.instruction()
    assert CorpusSearchProfile(**asdict(wider)) == wider
    assert RERANK_CANDIDATE_LIMITS[enhanced.passage_ranking] == 100
    assert RERANK_CANDIDATE_LIMITS[wider.passage_ranking] == 1000


@pytest.mark.parametrize("query_source", [OWNER_QUERY, PUBLIC_QUESTION_QUERY])
@pytest.mark.parametrize("ranking_policy", [MINILM_RANKING, MINILM_WIDE_RANKING])
def test_ranking_broker_delivers_only_public_passages_and_keeps_owner_answer(
    tmp_path, monkeypatch, query_source, ranking_policy
):
    settings = corpus(tmp_path)
    settings["public_profile"]["passage_ranking"] = ranking_policy
    settings["public_profile"]["reranker_query_source"] = query_source
    model_path = tmp_path / "frozen-relevance-encoder"
    model_path.mkdir()
    settings["reranker_path"] = str(model_path)
    observed_pairs = []
    original_search = passage_reranker.search_corpus
    searched_limits = []

    def search(*args, **kwargs):
        searched_limits.append(kwargs["limit"])
        return original_search(*args, **kwargs)

    monkeypatch.setattr(passage_reranker, "search_corpus", search)

    def tokenize(queries, passages, **kwargs):
        observed_pairs.extend(zip(queries, passages, strict=True))
        assert kwargs["max_length"] == 512
        return {
            "input_ids": torch.tensor([[int("Elmport" in passage)] for passage in passages]),
            "attention_mask": torch.ones((len(passages), 1), dtype=torch.int64),
        }

    def forward(**encoded):
        assert encoded["input_ids"].device.type == "cpu"
        assert not torch.is_grad_enabled()
        return SimpleNamespace(logits=encoded["input_ids"].float())

    monkeypatch.setattr(passage_reranker, "_cpu_encoder", lambda _: (tokenize, forward))
    entry = replace(
        PublicTaskView.from_record("nq", "nq-open", {"question": "Which fictional place?"}),
        input_profile=INPUT_PROFILE,
    )
    instance = runtime(
        tmp_path, entry, [call("corpus_search", query="fictional"), "Final answer: Elsewhere"]
    )
    instance.config["corpus_retrieval"] = {"nq-open": settings}
    arm = InferenceArm(
        "ranked",
        tool_call_mode=ToolCallMode.QWEN_XML,
        task_semantic_guidance=PUBLIC_TASK_SEMANTICS_V9,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(ood_retrieval, "_search_pool", lambda: pool)
        final = asyncio.run(instance.generate(entry, arm, "ranked"))
    assert final.text == "Elsewhere"  # Relevance never corrects the owner's wrong answer.
    assert final.intervention_counts["model_calls"] == 2
    assert final.intervention_counts["peer_model_calls"] == 0
    assert len(observed_pairs) == 2
    expected_query = (
        "Which fictional place?" if query_source == PUBLIC_QUESTION_QUERY else "fictional"
    )
    assert all(query == expected_query for query, _ in observed_pairs)
    assert all("Which fictional place?" not in passage for _, passage in observed_pairs)
    scope = ("ranked", arm.arm_id, entry.task_id)
    result = instance.journal.traces(scope, "corpus-search-result")[0]["result"]
    assert result["query"] == "fictional"  # Owner query is not rewritten for BM25.
    assert result["passages"][0]["passage_id"] == "1"
    requests = instance.journal.traces(scope, "rendered-request")
    assert "relevance model" in json.dumps(requests[0]["messages"])
    assert any(
        json.dumps(result, ensure_ascii=False) in m["content"] for m in requests[1]["messages"]
    )
    assert "candidate_pairs" not in json.dumps(requests)
    assert "reranker_path" not in json.dumps(requests)
    diagnostics = instance.journal.traces(scope, "corpus-rerank")[0]
    assert diagnostics["candidate_pairs"] == 2
    assert searched_limits == [RERANK_CANDIDATE_LIMITS[ranking_policy]]
    assert diagnostics["ranking_policy"] == ranking_policy
    assert diagnostics["candidate_limit"] == searched_limits[0]
    assert diagnostics["ranking_query_source"] == query_source
    assert diagnostics["ranking_query"] == expected_query
    assert diagnostics["input_tokens"] == 2
    reader = SimpleNamespace(
        get=lambda *args: final,
        model_outputs=lambda s: (),
        traces=lambda s, stage, **kwargs: (diagnostics,) if stage == "corpus-rerank" else (),
    )
    cost = trajectory_diagnostics(reader, (scope,), benchmark="nq-open")["retrieval_reranker_cost"]
    assert cost["candidate_pairs"] == 2
    assert cost["forward_batches"] == 1
    instance.journal.close()


def test_question_profile_does_not_fall_back_when_public_question_is_missing(tmp_path):
    settings = corpus(tmp_path)
    settings["public_profile"].update(
        passage_ranking=MINILM_RANKING, reranker_query_source=PUBLIC_QUESTION_QUERY
    )
    settings["reranker_path"] = str(tmp_path)
    journal = CandidateJournal(tmp_path / "missing-question.sqlite")
    with pytest.raises(ValueError):
        ood_retrieval.CorpusSearchSession(settings, journal, ("run", "arm", "nq"))
    journal.close()


def test_encoder_failure_does_not_silently_fall_back_to_a_different_condition(
    tmp_path, monkeypatch
):
    settings = corpus(tmp_path)
    settings["public_profile"]["passage_ranking"] = MINILM_RANKING
    settings["reranker_path"] = str(tmp_path)
    journal = CandidateJournal(tmp_path / "ranker-failure.sqlite")
    scope = ("failed", "ranked", "nq")
    session = ood_retrieval.CorpusSearchSession(settings, journal, scope)

    def unavailable(*args, **kwargs):
        raise RuntimeError("encoder unavailable")

    monkeypatch.setattr(passage_reranker, "search_ranked_corpus", unavailable)
    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(ood_retrieval, "_search_pool", lambda: pool)
        with pytest.raises(RuntimeError):
            asyncio.run(
                session.execute(
                    "fictional",
                    response=call("corpus_search", query="fictional"),
                    call_id="query1",
                    participant="owner",
                    final_ready=False,
                )
            )
    assert len(journal.traces(scope, "corpus-search-start")) == 1
    assert not journal.traces(scope, "corpus-search-result")
    assert not journal.traces(scope, "corpus-rerank")
    journal.close()
