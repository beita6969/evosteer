"""Published single-argument signatures can bind explicit literal arguments."""

import pytest

from skillev.evaluation.decision_transport import (
    ExplicitDecision,
    PublicSurface,
    normalize_decision,
)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ('Action: search[query]\nArguments: {"query":"blue shoes"}', "search[blue shoes]"),
        ('Action: 搜索[query]\nArguments: query: "蓝鞋"', "search[蓝鞋]"),
        ('Action: search[query]\nArgument: "blue shoes"', "search[blue shoes]"),
        ("Action: search[query]\nArgument: blue shoes", "search[blue shoes]"),
        ("Action: click[target]\ntarget: Blue XL", "click[Blue XL]"),
        ('Action: search\nQuery: "Blue shoes"', "search[Blue shoes]"),
        ("Action: click\nTARGET: Blue XL", "click[Blue XL]"),
        ('Action: 点击[target]\nArguments: {"target":"Blue XL"}', "click[Blue XL]"),
        (
            'Action: search[query]\n{"kind":"tool","resource_id":"webshop",'
            '"name":"search","arguments":{"query":"blue shoes"}}',
            "search[blue shoes]",
        ),
    ],
)
def test_explicit_signature_binds_only_the_supplied_argument(response, expected):
    surface = PublicSurface("webshop", 3, ("search", "click[Blue XL]"))
    result = normalize_decision(ExplicitDecision(response, 3), surface)
    assert result.action == expected
    assert result.error is None


@pytest.mark.parametrize(
    "response",
    [
        'Action: search[target]\nArguments: {"target":"blue shoes"}',
        'Action: search[query]\nArguments: {"target":"blue shoes"}',
        'Action: search[query]\nArguments: {"query":"blue shoes","extra":"value"}',
        'Action: click[Blue XL]\nArguments: {"target":"Other"}',
        'Action: click[target]\nArguments: {"target":"Not available"}',
        'Action: search[query]\nArguments: {"query":42}',
        'Action: search[query]\nArguments: query = str("blue shoes")',
        'Action: search[query]\n{"kind":"tool","resource_id":"webshop",'
        '"name":"click","arguments":{"target":"Blue XL"}}',
        'Action: search[query]\nArguments: {"query":"blue shoes"}\nAction: click[Other]',
    ],
)
def test_binding_does_not_overwrite_a_target_infer_an_argument_or_expand_permissions(response):
    surface = PublicSurface("webshop", 3, ("search", "click[Blue XL]", "click[Other]"))
    assert normalize_decision(ExplicitDecision(response, 3), surface).action is None
