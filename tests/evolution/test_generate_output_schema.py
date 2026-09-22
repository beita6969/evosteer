"""Generate authority is structural; authored prose remains the author's output."""

from copy import deepcopy
from dataclasses import replace

import pytest

from skillev.evolution import AuthoringFailedError, AuthoringResult, validate_authoring_result
from skillev.evolution.authoring import uncovered_edge_requirement_id
from skillev.evolution.external_sglang_authoring import authoring_result_schema
from skillev.runtime.skills import SkillRequirement
from tests.evolution.test_base_model_authoring_budget import _cases
from tests.v3_helpers import CharacterTokenizer

ESCAPED_PROSE_EXAMPLES = (
    "First line\nSecond line",
    'Use the public "quoted" instruction.',
    r"Inspect public C:\workspace\example paths.",
    "中文公开策略",
    '中文\n"quoted" and \\ backslash\tcombined',
    '"]}, {"drafts": [{"instructions": "still only text',
)


def generate_schema_cases():
    request, result, _ = _cases()[-1]
    second = replace(request.edge_exemplars[0], edge_id="synthetic-second-edge:1")
    edges = (*request.edge_exemplars, second)
    request = replace(
        request,
        edge_exemplars=edges,
        evidence=replace(request.evidence, importance_edge_ids=tuple(e.edge_id for e in edges)),
        authority=replace(request.authority, required_tools=("public-tool-a", "public-tool-b")),
    )
    draft = result.drafts[0]
    result = replace(
        result,
        drafts=(
            replace(
                draft,
                applicability=replace(
                    draft.applicability, required_tools=request.authority.required_tools
                ),
                requirements=(
                    *draft.requirements,
                    replace(
                        draft.requirements[0],
                        requirement_id=uncovered_edge_requirement_id(second.edge_id),
                    ),
                ),
            ),
        ),
    )
    good = result.to_value()
    bad = {}
    for field in ("contexts", "excluded_contexts", "required_tools", "task_families"):
        value = deepcopy(good)
        value["drafts"][0]["applicability"][field] = ["wrong-public-value"]
        bad[field] = value
    value = deepcopy(good)
    value["drafts"][0]["applicability"]["required_tools"].reverse()
    bad["tool-order"] = value
    for name, requirements in (
        ("requirements-missing", []),
        ("requirements-reordered", list(reversed(good["drafts"][0]["requirements"]))),
        ("requirements-duplicated", [good["drafts"][0]["requirements"][0]] * 2),
        ("requirements-extra", good["drafts"][0]["requirements"] * 2),
    ):
        value = deepcopy(good)
        value["drafts"][0]["requirements"] = requirements
        bad[name] = value
    for field in ("title", "summary", "instructions"):
        value = deepcopy(good)
        value["drafts"][0][field] = " \n\t"
        bad[field] = value
    for name, changes in (
        ("wrong-id", {"requirement_id": "unrelated-public-id"}),
        ("immutable", {"kind": "immutable-constraint"}),
        ("revision", {"replaces": ["old-strategy"], "change_reason": "revise"}),
        ("orphan-reason", {"change_reason": "unbound revision"}),
        ("blank-requirement", {"text": " \n\t"}),
    ):
        value = deepcopy(good)
        value["drafts"][0]["requirements"][0].update(changes)
        bad[name] = value
    value = deepcopy(good)
    del value["drafts"][0]["requirements"][0]["kind"]
    bad["missing-kind"] = value
    for name, count in (("missing-draft", 0), ("extra-draft", 2)):
        value = deepcopy(good)
        value["drafts"] *= count
        bad[name] = value
    return request, good, bad


def test_generate_structural_schema_copies_exact_authority_and_ordered_requirements() -> None:
    request, good, _ = generate_schema_cases()
    schema = authoring_result_schema(request)
    drafts = schema["properties"]["drafts"]
    assert drafts["minItems"] == drafts["maxItems"] == 1
    fields = drafts["items"]["properties"]
    assert fields["applicability"]["const"] == good["drafts"][0]["applicability"]
    requirements = fields["requirements"]
    assert requirements["minItems"] == requirements["maxItems"] == 2
    assert requirements["items"] is False
    assert [i["properties"]["requirement_id"]["const"] for i in requirements["prefixItems"]] == [
        i["requirement_id"] for i in good["drafts"][0]["requirements"]
    ]
    for item in requirements["prefixItems"]:
        assert "change_reason" not in item["properties"]
        assert item["additionalProperties"] is False
    no_tools = replace(request, authority=replace(request.authority, required_tools=()))
    no_tools_schema = authoring_result_schema(no_tools)
    assert (
        no_tools_schema["properties"]["drafts"]["items"]["properties"]["applicability"]["const"][
            "required_tools"
        ]
        == []
    )


def test_generate_structural_failures_remain_rejected_by_unchanged_admission() -> None:
    request, good, bad = generate_schema_cases()

    def admit(value):
        validate_authoring_result(
            request,
            AuthoringResult.from_value(value),
            tokenizer=CharacterTokenizer(),
            max_skill_instruction_tokens_per_draft=4096,
        )

    admit(good)
    for value in bad.values():
        with pytest.raises((AuthoringFailedError, ValueError)):
            admit(value)


@pytest.mark.parametrize("text", ESCAPED_PROSE_EXAMPLES)
def test_generate_escaped_prose_roundtrips_and_passes_unchanged_admission(text) -> None:
    request, value, _ = generate_schema_cases()
    for field in ("title", "summary", "instructions"):
        value["drafts"][0][field] = text
    for requirement in value["drafts"][0]["requirements"]:
        requirement["text"] = text
    result = AuthoringResult.from_value(value)
    assert result.to_value() == value
    validate_authoring_result(
        request,
        result,
        tokenizer=CharacterTokenizer(),
        max_skill_instruction_tokens_per_draft=4096,
    )


@pytest.mark.parametrize("text", ["", " \n\t"])
@pytest.mark.parametrize("field", ["title", "summary", "instructions", "requirement-text"])
def test_empty_or_blank_authored_prose_remains_rejected(text, field) -> None:
    _, value, _ = generate_schema_cases()
    if field == "requirement-text":
        value["drafts"][0]["requirements"][0]["text"] = text
    else:
        value["drafts"][0][field] = text
    with pytest.raises(ValueError):
        AuthoringResult.from_value(value)


def test_all_author_operations_use_escape_capable_prose_schema() -> None:
    for request, _, _ in _cases():
        fields = authoring_result_schema(request)["properties"]["drafts"]["items"]["properties"]
        for name in ("title", "summary", "instructions"):
            assert fields[name]["type"] == "string"
            assert "minLength" not in fields[name]
            assert "pattern" not in fields[name]
        requirements = fields["requirements"]
        items = requirements.get("prefixItems", [requirements["items"]])
        for item in items:
            for name in ("text", "change_reason"):
                if name in item["properties"]:
                    assert item["properties"][name]["type"] == "string"
                    assert "minLength" not in item["properties"][name]
                    assert "pattern" not in item["properties"][name]


@pytest.mark.parametrize("reason", ["", " \n\t"])
def test_revision_reason_remains_nonblank(reason) -> None:
    with pytest.raises(ValueError):
        SkillRequirement.from_value(
            {
                "requirement_id": "new-strategy",
                "text": "Public revision.",
                "kind": "evolvable-strategy",
                "replaces": ["old-strategy"],
                "change_reason": reason,
            }
        )
