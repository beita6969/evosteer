"""Literal owner queries, public source identities, CPU cost and no fallback answers."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from skillev_private.evaluation import dpr_retrieval, ood_retrieval

from skillev.evaluation.corpus_search import (
    CORPUS_ID,
    DPR_QUERY_POLICY,
    DPR_RANKING,
    DPR_SEARCH_PROFILE,
    INPUT_PROFILE,
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


def dense_settings(tmp_path):
    settings = corpus(tmp_path, corpus_id=CORPUS_ID)
    settings["public_profile"].update(
        profile_id=DPR_SEARCH_PROFILE, query_policy=DPR_QUERY_POLICY, passage_ranking=DPR_RANKING
    )
    settings.update(encoder_path=str(tmp_path), dense_index_path=str(tmp_path))
    return settings


def test_supervised_condition_is_explicit_and_not_a_bm25_default():
    legacy = CorpusSearchProfile()
    dense = replace(
        legacy,
        profile_id=DPR_SEARCH_PROFILE,
        query_policy=DPR_QUERY_POLICY,
        passage_ranking=DPR_RANKING,
    )
    assert "supervised on NQ training" in dense.instruction()
    assert "not mandatory phrase filters" in dense.instruction()
    assert "BM25" not in dense.instruction()
    assert CorpusSearchProfile(**asdict(dense)) == dense
    assert legacy.profile_id != dense.profile_id
    for field in ("profile_id", "query_policy", "passage_ranking"):
        with pytest.raises(ValueError):
            replace(dense, **{field: getattr(legacy, field)})


def test_dense_rows_keep_index_order_and_exact_source_text(tmp_path):
    settings = dense_settings(tmp_path)
    rows = dpr_retrieval.indexed_passages(Path(settings["index_path"]), [1, 0], [9.0, 8.0])
    assert [row["passage_id"] for row in rows] == ["2", "1"]
    assert [row["rank"] for row in rows] == [1, 2]
    assert rows[1]["text"] == "The fictional copper city is named Elmport."


@pytest.mark.parametrize(
    ("indices", "scores"), [([-1], [1.0]), ([2], [1.0]), ([0], []), ([0], [float("nan")])]
)
def test_missing_or_invalid_dense_result_is_not_an_empty_success(tmp_path, indices, scores):
    settings = dense_settings(tmp_path)
    with pytest.raises(ValueError):
        dpr_retrieval.indexed_passages(Path(settings["index_path"]), indices, scores)


def test_dense_broker_keeps_owner_answer_and_persists_cost_without_private_feedback(
    tmp_path, monkeypatch
):
    settings = dense_settings(tmp_path)
    seen = []

    def tokenize(query, **kwargs):
        seen.append(query)
        assert kwargs["max_length"] == 256
        return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}

    def forward(**inputs):
        assert inputs["input_ids"].device.type == "cpu"
        assert not torch.is_grad_enabled()
        return SimpleNamespace(pooler_output=torch.tensor([[3.0, 4.0]]))

    def search(vector, limit):
        assert vector.tolist() == [[3.0, 4.0]]  # No silent cosine normalization.
        return np.array([[9.0, 8.0]]), np.array([[1, 0]])

    monkeypatch.setattr(
        dpr_retrieval,
        "_cpu_retriever",
        lambda *args: (tokenize, forward, SimpleNamespace(search=search, ntotal=2), "test"),
    )
    entry = replace(
        PublicTaskView.from_record("nq", "nq-open", {"question": "Which fictional city?"}),
        input_profile=INPUT_PROFILE,
    )
    query = 'the "copper city"'
    instance = runtime(tmp_path, entry, [call("corpus_search", query=query), "Final answer: Wrong"])
    instance.config["corpus_retrieval"] = {"nq-open": settings}
    arm = InferenceArm(
        "dense",
        tool_call_mode=ToolCallMode.QWEN_XML,
        task_semantic_guidance=PUBLIC_TASK_SEMANTICS_V9,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(ood_retrieval, "_search_pool", lambda: pool)
        final = asyncio.run(instance.generate(entry, arm, "dense"))
    assert final.text == "Wrong"
    assert seen == [query]
    assert final.intervention_counts["model_calls"] == 2
    assert final.intervention_counts["peer_model_calls"] == 0
    scope = ("dense", arm.arm_id, entry.task_id)
    result = instance.journal.traces(scope, "corpus-search-result")[0]["result"]
    requests = instance.journal.traces(scope, "rendered-request")
    assert any(
        json.dumps(result, ensure_ascii=False) in m["content"] for m in requests[1]["messages"]
    )
    assert "inner_product_scores" not in json.dumps(requests)
    assert "encoder_path" not in json.dumps(requests)
    diagnostics = instance.journal.traces(scope, "corpus-dense-retrieval")[0]
    assert diagnostics["training_supervision"] == "NQ train"
    reader = SimpleNamespace(
        get=lambda *args: final,
        model_outputs=lambda scope: (),
        traces=lambda scope, stage, **kwargs: (diagnostics,)
        if stage == "corpus-dense-retrieval"
        else (),
    )
    cost = trajectory_diagnostics(reader, (scope,), benchmark="nq-open")["retrieval_dense_cost"]
    assert cost["input_tokens"] == 2
    assert cost["forward_batches"] == 1
    instance.journal.close()


def test_unavailable_encoder_does_not_fall_back_to_bm25_or_answer(tmp_path, monkeypatch):
    settings = dense_settings(tmp_path)
    journal = CandidateJournal(tmp_path / "failure.sqlite")
    scope = ("failed", "dense", "nq")
    session = ood_retrieval.CorpusSearchSession(settings, journal, scope)

    def fail(*args, **kwargs):
        raise RuntimeError("unavailable encoder")

    monkeypatch.setattr(dpr_retrieval, "search_dense_corpus", fail)
    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(ood_retrieval, "_search_pool", lambda: pool)
        with pytest.raises(RuntimeError):
            asyncio.run(
                session.execute(
                    "query",
                    response=call("corpus_search", query="query"),
                    call_id="c1",
                    participant="owner",
                    final_ready=False,
                )
            )
    assert not journal.traces(scope, "corpus-search-result")
    journal.close()
