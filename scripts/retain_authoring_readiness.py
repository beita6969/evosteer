#!/usr/bin/env python3
"""One-call-per-case real-9B Retain development or untouched holdout run."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import cast

from scripts.real_9b_gate_4a import (
    RecordingAuthoringBackbone,
    _backbone_config,
    _failure_chain,
    _generation_evidence,
    _hardware_identity,
)
from skillev.contracts import JsonValue, canonical_json, stable_hash
from skillev.evolution import (
    AUTHORING_TEMPLATE_RETAIN,
    RETAIN_AUTHORING_SAMPLING,
    AuthoringCallMaximum,
    AuthoringFailedError,
    BaseModelSkillAuthor,
    EvolutionConfig,
    RetainReadinessCase,
    RetainSuiteKind,
    retain_compression_budget,
    retain_readiness_suite,
)
from skillev.experiments.build_identity import (
    require_source_archive_matches_execution,
    sha256_file,
)
from skillev.policy import (
    BaseModelArtifactIdentity,
    PolicyBackbone,
    QwenMultimodalPolicyBackbone,
    QwenTokenizerAdapter,
)
from skillev.runtime import (
    BudgetLedger,
    BudgetVector,
    LiveAttemptEventLog,
    RuntimeEventEmitter,
)

RETAIN_READINESS_RESULT_FORMAT = "skillev-real-9b-retain-readiness@2"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--eos-token-id", type=int, required=True)
    parser.add_argument("--source-package", required=True)
    parser.add_argument("--work-directory", required=True)
    parser.add_argument(
        "--suite",
        choices=tuple(item.value for item in RetainSuiteKind),
        required=True,
    )
    return parser.parse_args()


def _case_record(
    *,
    case: RetainReadinessCase,
    author: BaseModelSkillAuthor,
    recording: RecordingAuthoringBackbone,
    tokenizer: QwenTokenizerAdapter,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    request = case.request
    budget = retain_compression_budget(request.source, tokenizer=tokenizer)
    results_before = len(recording.results)
    started = time.monotonic()
    error_chain: list[JsonValue] = []
    classification = "validated"
    result_hash: str | None = None
    output_counts: list[JsonValue] = []
    try:
        authored = author.author(request)
        result_hash = stable_hash(authored.to_value())
        output_counts = [
            len(tokenizer.encode(canonical_json(draft.to_value()))) for draft in authored.drafts
        ]
    except AuthoringFailedError as error:
        classification = "rejected"
        error_chain = _failure_chain(error)
    generated = recording.results[results_before:]
    generation: JsonValue = None
    if len(generated) == 1:
        generation = _generation_evidence(tokenizer=tokenizer, result=generated[0])
    elif generated:
        raise RuntimeError("one Retain readiness case made more than one model call")
    public = {
        "budget": budget.to_value(),
        "case_id": case.case_id,
        "classification": classification,
        "elapsed_seconds": time.monotonic() - started,
        "error_chain": error_chain,
        "generation": generation,
        "output_model_visible_token_counts": output_counts,
        "profiles": list(case.profiles),
        "reservation_id": f"phi-authoring:{AUTHORING_TEMPLATE_RETAIN}:{request.seed}",
        "result_hash": result_hash,
        "source_content_hash": request.source.manifest.content_hash,
    }
    private = {
        "case_id": case.case_id,
        "decoded_generation": (
            tokenizer.decode(generated[0].content_token_ids) if len(generated) == 1 else None
        ),
        "error_chain": error_chain,
    }
    return public, private


def main() -> None:
    args = _args()
    started = time.monotonic()
    work = Path(args.work_directory)
    work.mkdir(parents=True, exist_ok=False)
    source_package = Path(args.source_package)
    if not source_package.is_file():
        raise FileNotFoundError(source_package)
    executing_root = Path(__file__).resolve().parents[1]
    source_provenance = require_source_archive_matches_execution(
        source_archive=source_package,
        executing_root=executing_root,
        temporary_parent=work,
    )
    suite = retain_readiness_suite(RetainSuiteKind(args.suite))
    hardware = _hardware_identity()
    config = _backbone_config(args)
    backbone = QwenMultimodalPolicyBackbone(config)
    tokenizer = cast(QwenTokenizerAdapter, backbone.tokenizer)
    recording = RecordingAuthoringBackbone(backbone)
    evolution = EvolutionConfig(generate_min_absolute_log_importance=0.1)
    maximum = AuthoringCallMaximum(
        input_tokens=evolution.max_authoring_prompt_tokens,
        output_tokens=evolution.max_authoring_completion_tokens,
    )
    ledger = BudgetLedger(
        run_id=f"retain-readiness-{suite.kind.value}",
        attempt_id=suite.version.replace("@", "-"),
        cap=BudgetVector(
            input_tokens=len(suite.cases) * maximum.input_tokens,
            output_tokens=len(suite.cases) * maximum.output_tokens,
            model_calls=len(suite.cases),
        ),
    )
    events = LiveAttemptEventLog(
        work / "events.jsonl",
        run_id=ledger.run_id,
        attempt_id=ledger.attempt_id,
    )
    author = BaseModelSkillAuthor(
        backbone=cast(PolicyBackbone, recording),
        config=evolution,
        sampling=RETAIN_AUTHORING_SAMPLING,
        ledger=ledger,
        emitter=RuntimeEventEmitter(events, producer_id="retain-readiness"),
        maximum=maximum,
    )
    paired_records = [
        _case_record(
            case=case,
            author=author,
            recording=recording,
            tokenizer=tokenizer,
        )
        for case in suite.cases
    ]
    records = [item[0] for item in paired_records]
    private_records = [item[1] for item in paired_records]
    if len(recording.requests) != len(suite.cases):
        raise RuntimeError("every feasible Retain readiness case must make exactly one call")
    ledger.assert_fully_settled()
    passed = all(item["classification"] == "validated" for item in records)
    result: dict[str, JsonValue] = {
        "base_model_artifact_content_hash": cast(
            BaseModelArtifactIdentity,
            config.base_model_artifact,
        ).content_hash,
        "elapsed_seconds": time.monotonic() - started,
        "format": RETAIN_READINESS_RESULT_FORMAT,
        "hardware": hardware.to_value(),
        "hardware_content_hash": hardware.content_hash,
        "model_call_count": len(recording.requests),
        "passed": passed,
        "records": records,
        "sampling": RETAIN_AUTHORING_SAMPLING.to_value(),
        "source_package_sha256": sha256_file(source_package),
        "source_provenance": source_provenance.to_value(),
        "suite_content_hash": suite.content_hash,
        "suite_kind": suite.kind.value,
        "suite_version": suite.version,
        "template_version": AUTHORING_TEMPLATE_RETAIN,
        "tokenizer_identity": tokenizer.public_identity.to_value(),
    }
    (work / "retain-readiness-result.json").write_text(
        canonical_json(result) + "\n",
        encoding="utf-8",
    )
    private_output = work / "retain-readiness-private.json"
    private_output.write_text(
        canonical_json({"records": private_records}) + "\n",
        encoding="utf-8",
    )
    private_output.chmod(0o600)
    if not passed:
        raise RuntimeError("one or more one-shot Retain readiness cases failed")


if __name__ == "__main__":
    main()
