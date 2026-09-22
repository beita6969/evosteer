"""Answer-free aggregate diagnostics; no model calls, score changes or candidate selection."""

from __future__ import annotations

import ast
from collections import Counter
from typing import Any, cast

from .sealed_candidates import CandidateReader, EventOrigin


def _events(
    reader: CandidateReader, scope: tuple[str, str, str], stage: str, *, origin: EventOrigin
) -> tuple[dict[str, Any], ...]:
    rows = reader.traces(scope, stage, origin=origin)
    if any(not isinstance(row, dict) for row in rows):
        raise RuntimeError("trajectory diagnostic evidence has an invalid record shape")
    return tuple(cast(dict[str, Any], row) for row in rows)


def trajectory_diagnostics(
    reader: CandidateReader,
    scopes: tuple[tuple[str, str, str], ...],
    *,
    benchmark: str,
) -> dict[str, object]:
    counts: Counter[str] = Counter()
    terminal: Counter[str] = Counter()
    submissions: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    judges: Counter[str] = Counter()
    composition: dict[str, Counter[str]] = {
        key: Counter()
        for key in ("difficulty", "domain", "language", "specialty", "department", "snapshot")
    }
    rubric: Counter[str] = Counter()
    retrieval_cost: Counter[str] = Counter()
    retrieval_seconds = 0.0
    rerankers: Counter[str] = Counter()
    dense_cost: Counter[str] = Counter()
    dense_seconds = 0.0
    dense_encoders: Counter[str] = Counter()
    hybrid_cost: Counter[str] = Counter()
    hybrid_seconds = 0.0
    hybrid_rankings: Counter[str] = Counter()
    retrieval_queue_seconds = 0.0
    retrieval_workers_started = 0
    for scope in scopes:
        candidate = reader.get(*scope)
        counts["planned"] += 1
        counts["empty_submissions"] += int(not candidate.text.strip())
        counts["whole_text_unlabelled"] += int(
            "unlabelled" in (candidate.submission or {}).get("projection_id", "")
        )
        counts["submitted_values_over_eight_words"] += int(len(candidate.text.split()) > 8)
        terminal[candidate.terminal_status.value] += 1
        outputs = reader.model_outputs(scope)
        counts["recorded_owner_calls"] += len(outputs)
        counts["owner_output_tokens"] += sum(
            row.result["usage"]["output_tokens"] for row in outputs
        )
        counts["consultant_calls"] += sum(row.participant != "owner" for row in outputs)
        if candidate.text and benchmark in {"livecodebench", "apps-introductory"}:
            try:
                ast.parse(candidate.text)
            except SyntaxError:
                counts["submitted_python_syntax_errors"] += 1
        identities = _events(reader, scope, "source-identity", origin=EventOrigin.MODEL_TRANSPORT)
        counts["missing_source_identity"] += int(not identities)
        counts["missing_source_revision"] += int(
            not identities or not identities[0].get("source_revision")
        )
        receipts = _events(
            reader, scope, "public-input-receipt", origin=EventOrigin.MODEL_TRANSPORT
        )
        counts["missing_input_receipt"] += int(not receipts)
        for row in receipts:
            counts["incomplete_required_inputs"] += int(row["required_input_received"] is False)
            counts["out_of_order_public_paragraph_calls"] += int(
                row.get("paragraph_order_received") is False
            )
            counts["unverified_source_exports"] += int(row["exported_source_complete"] is None)
            counts["incomplete_source_exports"] += int(row["exported_source_complete"] is False)
        endings = _events(reader, scope, "submission-outcome", origin=EventOrigin.MODEL_TRANSPORT)
        submissions.update(row.get("preseal_state") or row["state"] for row in endings)
        counts["missing_submission_state"] += int(not endings)
        transitions.update(
            row["category"]
            for row in _events(
                reader, scope, "environment-transition", origin=EventOrigin.ENVIRONMENT
            )
        )
        counts["unparsed_control_messages_not_delivered"] += len(
            _events(reader, scope, "control-parse-failure", origin=EventOrigin.ACTOR_DIAGNOSTIC)
        )
        for row in _events(reader, scope, "corpus-rerank", origin=EventOrigin.ENVIRONMENT):
            for key in ("candidate_pairs", "input_tokens", "forward_batches"):
                retrieval_cost[key] += row[key]
            retrieval_seconds += row["rerank_seconds"]
            rerankers[row["model_id"]] += 1
        for row in _events(reader, scope, "corpus-dense-retrieval", origin=EventOrigin.ENVIRONMENT):
            for key in ("input_tokens", "forward_batches"):
                dense_cost[key] += row[key]
            dense_seconds += row["search_seconds"]
            dense_encoders[row["model_id"]] += 1
        for row in _events(
            reader, scope, "corpus-hybrid-retrieval", origin=EventOrigin.ENVIRONMENT
        ):
            for key in ("dense_candidates", "lexical_candidates", "union_candidates"):
                hybrid_cost[key] += row[key]
            hybrid_seconds += row["lexical_seconds"] + row["fusion_seconds"]
            hybrid_rankings[row["ranking_policy"]] += 1
        for row in _events(
            reader, scope, "corpus-search-worker-start", origin=EventOrigin.ENVIRONMENT
        ):
            retrieval_queue_seconds += row["queue_seconds"]
            retrieval_workers_started += 1
        for row in _events(reader, scope, "native-ood-verdict", origin=EventOrigin.SCORER):
            if row.get("failure_kind"):
                failures[row["failure_kind"]] += 1
            judges.update(model or "unknown" for model in row.get("returned_judge_models", []))
            for key in composition:
                value = row.get("source_metadata", {}).get(key, row.get(key))
                if isinstance(value, list):
                    composition[key].update(str(item) for item in value)
                else:
                    composition[key][str(value) if value is not None else "unknown"] += 1
            for key in (
                "positive_criteria_count",
                "positive_criteria_met",
                "positive_denominator",
                "positive_earned_points",
                "triggered_negative_count",
                "negative_triggered_points",
            ):
                if key in row:
                    rubric[key] += row[key]
    return {
        "counts": dict(counts),
        "candidate_statuses": dict(terminal),
        "submission_states": dict(submissions),
        "environment_transition_categories": dict(transitions),
        "native_code_failure_kinds": dict(failures),
        "returned_judge_models": dict(judges),
        "reported_source_composition": {key: dict(values) for key, values in composition.items()},
        "rubric_coverage": dict(rubric),
        "retrieval_reranker_cost": {
            **dict(retrieval_cost),
            "rerank_worker_seconds": retrieval_seconds,
            "queries_by_model": dict(rerankers),
            "role": "public-passage-relevance-only-not-owner-answering",
        },
        "retrieval_dense_cost": {
            **dict(dense_cost),
            "search_worker_seconds": dense_seconds,
            "queries_by_model": dict(dense_encoders),
            "role": "NQ-supervised-public-passage-retrieval-only-not-owner-answering",
        },
        "retrieval_hybrid_cost": {
            **dict(hybrid_cost),
            "lexical_and_fusion_worker_seconds": hybrid_seconds,
            "queries_by_ranking": dict(hybrid_rankings),
            "dense_cost_reported_separately": True,
            "role": "public-passage-rank-fusion-only-not-owner-answering",
        },
        "retrieval_cpu_queue": {
            "queries_admitted": retrieval_workers_started,
            "summed_queue_seconds": retrieval_queue_seconds,
        },
        "interpretation": (
            "word counts and syntax status are diagnostics, "
            "not answer selectors or substitute scores"
        ),
    }
