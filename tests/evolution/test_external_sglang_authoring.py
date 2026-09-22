from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from skillev.contracts import JsonValue, canonical_json
from skillev.evolution import (
    AuthoringCallMaximum,
    AuthoringFailedError,
    AuthoringRequest,
    AuthoringSamplingConfig,
    EvolutionConfig,
    ExternalSGLangSkillAuthor,
    render_authoring_prompt,
    validate_authoring_result,
)
from skillev.evolution.external_sglang_authoring import authoring_result_schema
from skillev.rollout.external_sglang import ExternalSGLangRolloutConfig
from skillev.runtime import BudgetLedger, LiveAttemptEventLog, RuntimeEventEmitter
from skillev.runtime.request_journal import DurableRequestJournal
from skillev.runtime.sglang_gateway import SGLangGateway, SGLangGatewayConfig
from skillev.runtime.skills import SkillRequirement
from tests.evolution.test_base_model_authoring_budget import _cases
from tests.v3_helpers import CharacterTokenizer


class _Transport:
    def __init__(self, *, tokenizer: CharacterTokenizer, content: str) -> None:
        self.tokenizer = tokenizer
        self.content = content
        self.payload: dict[str, JsonValue] | None = None
        self.calls = 0

    def request(
        self,
        *,
        method: str,
        url: str,
        payload,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> tuple[int, JsonValue]:
        del method, url, timeout_seconds, max_response_bytes
        self.calls += 1
        self.payload = dict(payload)
        content_ids = list(self.tokenizer.encode(self.content))
        prompt_ids = cast(list[int], self.payload["input_ids"])
        return 200, {
            "meta_info": {
                "completion_tokens": len(content_ids) + 1,
                "finish_reason": {"matched": 0, "type": "stop"},
                "prompt_tokens": len(prompt_ids),
            },
            "output_ids": [999, *content_ids, 0],
            "text": self.content,
        }


def test_external_sglang_authoring_uses_adapter_free_exact_tokens(tmp_path: Path) -> None:
    request, expected, _ = _cases()[-1]
    tokenizer = CharacterTokenizer()
    transport = _Transport(tokenizer=tokenizer, content=canonical_json(expected.to_value()))
    maximum = AuthoringCallMaximum(input_tokens=100_000, output_tokens=100_000)
    ledger = BudgetLedger(
        run_id="external-authoring",
        attempt_id="attempt-1",
        cap=maximum.to_budget_vector(),
    )
    gateway = SGLangGateway(
        SGLangGatewayConfig(
            endpoint_base="http://127.0.0.1:30000",
            base_model="qwen-base",
            supervisor_adapter="forward-adapter",
        )
    )
    author = ExternalSGLangSkillAuthor(
        tokenizer=tokenizer,
        config=EvolutionConfig(
            generate_min_absolute_log_importance=0.1,
            max_skill_instruction_tokens_per_draft=4096,
            max_authoring_completion_tokens=maximum.output_tokens,
            max_authoring_prompt_tokens=maximum.input_tokens,
        ),
        sampling=AuthoringSamplingConfig(temperature=1.0, top_p=1.0),
        ledger=ledger,
        emitter=RuntimeEventEmitter(
            LiveAttemptEventLog(
                tmp_path / "events.jsonl",
                run_id=ledger.run_id,
                attempt_id=ledger.attempt_id,
            ),
            producer_id="external-author",
        ),
        maximum=maximum,
        rollout_config=ExternalSGLangRolloutConfig(endpoint_base="http://127.0.0.1:30000"),
        gateway=gateway,
        transport=transport,
        request_journal=DurableRequestJournal(tmp_path / "requests.sqlite3"),
    )

    assert author.author(cast(AuthoringRequest, request)) == expected
    assert transport.payload is not None
    assert "lora_path" not in transport.payload
    sampling = cast(dict[str, JsonValue], transport.payload["sampling_params"])
    assert sampling["sampling_seed"] == request.seed
    assert isinstance(sampling["json_schema"], str)
    assert json.loads(sampling["json_schema"]) == authoring_result_schema(request)
    ledger.assert_fully_settled()

    restored_ledger = BudgetLedger(
        run_id=ledger.run_id, attempt_id="attempt-2", cap=maximum.to_budget_vector()
    )
    restored = replace(
        author,
        ledger=restored_ledger,
        emitter=RuntimeEventEmitter(
            LiveAttemptEventLog(
                tmp_path / "restored.jsonl",
                run_id=ledger.run_id,
                attempt_id=restored_ledger.attempt_id,
            ),
            producer_id="external-author",
        ),
        request_journal=DurableRequestJournal(tmp_path / "requests.sqlite3"),
    )
    assert restored.author(cast(AuthoringRequest, request)) == expected
    assert transport.calls == 1
    restored_ledger.assert_fully_settled()
    assert restored_ledger.settled == ledger.settled


@pytest.mark.parametrize("kind", [None, "immutable-constraint", "evolvable-strategy"])
def test_generate_schema_and_admission_agree_on_new_requirement_kind(kind) -> None:
    request, result, _ = _cases()[-1]
    schema = authoring_result_schema(request)
    requirement_schema = schema["properties"]["drafts"]["items"]["properties"]["requirements"][
        "prefixItems"
    ][0]
    raw = result.drafts[0].requirements[0].to_value()
    if kind is None:
        raw.pop("kind")
    else:
        raw["kind"] = kind
    allowed = all(key in raw for key in requirement_schema["required"]) and (
        raw.get("kind") in requirement_schema["properties"]["kind"]["enum"]
    )
    assert allowed is (kind == "evolvable-strategy")
    requirement = SkillRequirement.from_value(raw)
    draft = replace(result.drafts[0], requirements=(requirement,))
    changed = replace(result, drafts=(draft,))
    if allowed:
        validate_authoring_result(
            request,
            changed,
            tokenizer=CharacterTokenizer(),
            max_skill_instruction_tokens_per_draft=4096,
        )
    else:
        # Missing kind still decodes as historical immutable authority, not a patch.
        assert requirement.immutable
        with pytest.raises(AuthoringFailedError):
            validate_authoring_result(
                request,
                changed,
                tokenizer=CharacterTokenizer(),
                max_skill_instruction_tokens_per_draft=4096,
            )


def test_generate_schema_is_operation_specific_and_prompt_agrees() -> None:
    cases = _cases()
    request, result, _ = cases[-1]
    schema = authoring_result_schema(request)
    item = schema["properties"]["drafts"]["items"]["properties"]["requirements"]["prefixItems"][0]
    assert item["properties"]["replaces"]["maxItems"] == 0
    prompt = json.loads(
        render_authoring_prompt(request, tokenizer=CharacterTokenizer()).split("\n")[1]
    )
    assert "kind" in prompt["output_contract"]["requirement_fields"]
    assert "kind" not in prompt["output_contract"]["optional_requirement_fields"]
    revision = replace(
        result.drafts[0].requirements[0],
        replaces=("synthetic-prior-strategy",),
        change_reason="A Generate draft cannot revise an existing strategy.",
    )
    with pytest.raises(AuthoringFailedError):
        validate_authoring_result(
            request,
            replace(result, drafts=(replace(result.drafts[0], requirements=(revision,)),)),
            tokenizer=CharacterTokenizer(),
            max_skill_instruction_tokens_per_draft=4096,
        )
    for other, _, _ in cases[:-1]:
        other_schema = authoring_result_schema(other)
        other_item = other_schema["properties"]["drafts"]["items"]["properties"]["requirements"][
            "items"
        ]
        assert "kind" not in other_item["required"]
        assert "immutable-constraint" in other_item["properties"]["kind"]["enum"]
        assert "maxItems" not in other_item["properties"]["replaces"]
