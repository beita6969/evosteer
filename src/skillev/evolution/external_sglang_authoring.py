"""Adapter-free exact-token skill authoring through native SGLang generation."""

from __future__ import annotations

import json
import unicodedata
from copy import deepcopy
from dataclasses import dataclass, field
from typing import cast

from skillev.contracts import JsonValue, canonical_json, normalize_json
from skillev.policy import AuthoringTokenizerProtocol
from skillev.rollout.external_sglang import ExternalSGLangRolloutConfig
from skillev.runtime import (
    BudgetLedger,
    BudgetReservation,
    BudgetSettlement,
    BudgetVector,
    EventType,
    RuntimeEventEmitter,
)
from skillev.runtime.request_journal import DurableRequestJournal
from skillev.runtime.sglang_gateway import (
    SGLangControlTransport,
    SGLangGateway,
    SGLangGatewayConfig,
    SGLangGatewayError,
    UrllibSGLangControlTransport,
)

from .authoring import (
    AuthoringCallMaximum,
    AuthoringFailedError,
    AuthoringRequest,
    AuthoringResult,
    AuthoringSamplingConfig,
    GenerateAuthoringRequest,
    RetainAuthoringRequest,
    authoring_reservation_id,
    generate_output_constraints,
    retain_generation_token_limit,
    validate_authoring_result,
)
from .config import EvolutionConfig

# xgrammar 0.2.1 lowers minLength strings to a character class that excludes
# JSON escapes. Keep prose on its standard JSON-string grammar; the unchanged
# draft/requirement admission enforces nonempty, non-whitespace content.
_AUTHORING_RESULT_SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["drafts"],
    "properties": {
        "drafts": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "applicability",
                    "instructions",
                    "requirements",
                    "summary",
                    "title",
                ],
                "properties": {
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "instructions": {"type": "string"},
                    "applicability": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "contexts",
                            "excluded_contexts",
                            "required_tools",
                            "task_families",
                        ],
                        "properties": {
                            "contexts": {"type": "array", "items": {"type": "string"}},
                            "excluded_contexts": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "required_tools": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "task_families": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                    "requirements": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["requirement_id", "text"],
                            "properties": {
                                "requirement_id": {"type": "string", "minLength": 1},
                                "text": {"type": "string"},
                                "kind": {
                                    "type": "string",
                                    "enum": ["immutable-constraint", "evolvable-strategy"],
                                },
                                "replaces": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "uniqueItems": True,
                                },
                                "change_reason": {"type": "string"},
                            },
                        },
                    },
                },
            },
        }
    },
}


def authoring_result_schema(request: AuthoringRequest) -> dict[str, JsonValue]:
    """Constrain new Generate strategies without reinterpreting historical drafts.

    Other operations retain their source-copy/revision schema. Their immutable
    authority rules, like Generate admission, remain enforced by the validator.
    """
    schema = deepcopy(_AUTHORING_RESULT_SCHEMA)
    if isinstance(request, GenerateAuthoringRequest):
        properties = cast(dict[str, JsonValue], schema["properties"])
        drafts = cast(dict[str, JsonValue], properties["drafts"])
        drafts["maxItems"] = 1
        draft = cast(dict[str, JsonValue], drafts["items"])
        fields = cast(dict[str, JsonValue], draft["properties"])
        constraints = generate_output_constraints(request)
        applicability = cast(list[JsonValue], constraints["applicability_by_draft"])[0]
        fields["applicability"] = {"const": applicability}
        requirements = cast(dict[str, JsonValue], fields["requirements"])
        requirement = cast(dict[str, JsonValue], requirements["items"])
        requirement["required"] = ["requirement_id", "text", "kind"]
        requirement_fields = cast(dict[str, JsonValue], requirement["properties"])
        requirement_fields["kind"] = {"type": "string", "enum": ["evolvable-strategy"]}
        replaces = cast(dict[str, JsonValue], requirement_fields["replaces"])
        replaces["maxItems"] = 0
        del requirement_fields["change_reason"]
        ids = cast(list[JsonValue], constraints["required_requirement_ids_by_draft"])[0]
        assert isinstance(ids, list)
        ordered: list[JsonValue] = []
        for requirement_id in ids:
            item = deepcopy(requirement)
            cast(dict[str, JsonValue], item["properties"])["requirement_id"] = {
                "const": requirement_id
            }
            ordered.append(item)
        requirements["prefixItems"] = ordered
        requirements["items"] = False
        requirements["minItems"] = len(ordered)
        requirements["maxItems"] = len(ordered)
    return schema


@dataclass(slots=True)
class ExternalSGLangSkillAuthor:
    """Run Phi authoring on the shared frozen base without any LoRA adapter."""

    tokenizer: AuthoringTokenizerProtocol
    config: EvolutionConfig
    sampling: AuthoringSamplingConfig
    ledger: BudgetLedger
    emitter: RuntimeEventEmitter
    maximum: AuthoringCallMaximum
    rollout_config: ExternalSGLangRolloutConfig
    gateway: SGLangGateway
    replica_configs: tuple[SGLangGatewayConfig, ...] = ()
    request_journal: DurableRequestJournal | None = None
    transport: SGLangControlTransport = field(
        default_factory=UrllibSGLangControlTransport,
        repr=False,
    )

    def __post_init__(self) -> None:
        if any(v.base_model != self.gateway.config.base_model for v in self.replica_configs):
            raise ValueError("author replicas must use the same frozen base model")
        if self.maximum.input_tokens != self.config.max_authoring_prompt_tokens:
            raise ValueError("authoring input maximum differs from EvolutionConfig")
        if self.maximum.output_tokens != self.config.max_authoring_completion_tokens:
            raise ValueError("authoring output maximum differs from EvolutionConfig")

    def author(self, request: AuthoringRequest) -> AuthoringResult:
        from .authoring_material import render_bounded_authoring_prompt

        prompt = render_bounded_authoring_prompt(
            request, tokenizer=self.tokenizer, maximum_tokens=self.maximum.input_tokens
        )
        input_ids = tuple(self.tokenizer.encode_authoring_prompt(prompt))
        if not input_ids:
            raise AuthoringFailedError("authoring prompt encoded to no tokens")
        if len(input_ids) > self.maximum.input_tokens:
            raise AuthoringFailedError("authoring prompt exceeds its fixed maximum")
        output_maximum = self.maximum.output_tokens
        if isinstance(request, RetainAuthoringRequest):
            output_maximum = min(
                output_maximum,
                retain_generation_token_limit(request.source, tokenizer=self.tokenizer),
            )
        reservation = BudgetReservation(
            reservation_id=authoring_reservation_id(request),
            run_id=self.ledger.run_id,
            attempt_id=self.ledger.attempt_id,
            invocation_id=f"phi-authoring:{request.seed}",
            maximum=BudgetVector(
                input_tokens=len(input_ids),
                output_tokens=output_maximum,
                model_calls=1,
            ),
        )
        self.ledger.reserve(reservation)
        self.emitter.emit(
            EventType.BUDGET_RESERVED,
            {
                "maximum": reservation.maximum.to_value(),
                "reservation_id": reservation.reservation_id,
            },
        )
        payload = cast(
            dict[str, JsonValue],
            normalize_json(
                {
                    "input_ids": list(input_ids),
                    "sampling_params": {
                        "json_schema": canonical_json(authoring_result_schema(request)),
                        "max_new_tokens": output_maximum,
                        "sampling_seed": request.seed,
                        "temperature": self.sampling.temperature,
                        "top_p": self.sampling.top_p,
                    },
                    "stream": False,
                }
            ),
        )
        gateway = (
            SGLangGateway(self.replica_configs[request.seed % len(self.replica_configs)])
            if self.replica_configs
            else self.gateway
        )
        gateway.begin_skill_creator_request()
        try:
            endpoint = (
                gateway.config.api_root + "/generate"
                if self.replica_configs
                else self.rollout_config.generate_url
            )

            def send() -> tuple[int, JsonValue]:
                return self.transport.request(
                    method="POST",
                    url=endpoint,
                    payload=payload,
                    timeout_seconds=float(self.rollout_config.request_timeout_seconds),
                    max_response_bytes=self.rollout_config.max_response_bytes,
                )

            status, raw = (
                send()
                if self.request_journal is None
                else self.request_journal.request(
                    identity=("author", self.ledger.run_id, reservation.reservation_id),
                    endpoint=endpoint,
                    payload=payload,
                    send=send,
                )
            )
        except SGLangGatewayError as error:
            raise AuthoringFailedError("SGLang base-model authoring request failed") from error
        finally:
            gateway.end_skill_creator_request()
        if status != 200:
            raise AuthoringFailedError("SGLang base-model authoring returned failure")
        content_ids, stop_ids, text, prompt_tokens = self._parse(raw)
        if prompt_tokens != len(input_ids):
            raise AuthoringFailedError("SGLang authoring prompt usage differs from input IDs")
        if _normalized_text(self.tokenizer.decode(content_ids)) != _normalized_text(text):
            raise AuthoringFailedError("SGLang authoring text differs from output IDs")
        actual_usage = BudgetVector(
            input_tokens=prompt_tokens,
            output_tokens=len(content_ids) + len(stop_ids),
            model_calls=1,
        )
        self.ledger.settle(
            BudgetSettlement(
                reservation_id=reservation.reservation_id,
                actual=actual_usage,
            )
        )
        self.emitter.emit(
            EventType.BUDGET_SETTLED,
            {
                "actual": actual_usage.to_value(),
                "reservation_id": reservation.reservation_id,
            },
        )
        if not content_ids:
            raise AuthoringFailedError("SGLang frozen base returned no content")
        try:
            authored = AuthoringResult.from_value(json.loads(self.tokenizer.decode(content_ids)))
            validate_authoring_result(
                request,
                authored,
                tokenizer=self.tokenizer,
                max_skill_instruction_tokens_per_draft=(
                    self.config.max_skill_instruction_tokens_per_draft
                ),
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise AuthoringFailedError("authoring output violates its sealed schema") from error
        return authored

    @staticmethod
    def _parse(raw: JsonValue) -> tuple[tuple[int, ...], tuple[int, ...], str, int]:
        if not isinstance(raw, dict):
            raise AuthoringFailedError("SGLang authoring response must be an object")
        output_ids = raw.get("output_ids")
        text = raw.get("text")
        meta = raw.get("meta_info")
        if (
            not isinstance(output_ids, list)
            or not isinstance(text, str)
            or not isinstance(meta, dict)
        ):
            raise AuthoringFailedError("SGLang authoring response fields are incomplete")
        completion_tokens = meta.get("completion_tokens")
        prompt_tokens = meta.get("prompt_tokens")
        finish = meta.get("finish_reason")
        if (
            type(completion_tokens) is not int
            or completion_tokens < 0
            or type(prompt_tokens) is not int
            or prompt_tokens < 0
            or not isinstance(finish, dict)
            or not isinstance(finish.get("type"), str)
        ):
            raise AuthoringFailedError("SGLang authoring usage or finish reason is invalid")
        if completion_tokens > len(output_ids):
            raise AuthoringFailedError("SGLang authoring count exceeds output IDs")
        generated = output_ids[-completion_tokens:] if completion_tokens else []
        if any(type(token_id) is not int or token_id < 0 for token_id in generated):
            raise AuthoringFailedError("SGLang authoring output IDs are invalid")
        generated_ids = tuple(cast(list[int], generated))
        matched = finish.get("matched")
        if type(matched) is int and generated_ids and generated_ids[-1] == matched:
            return generated_ids[:-1], (matched,), text, prompt_tokens
        return generated_ids, (), text, prompt_tokens


def _normalized_text(value: str) -> str:
    return unicodedata.normalize("NFC", value)


__all__ = ["ExternalSGLangSkillAuthor"]
