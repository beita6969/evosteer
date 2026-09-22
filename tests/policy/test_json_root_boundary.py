from __future__ import annotations

import pytest

from skillev.policy.json_root_boundary import authoring_json_root_is_complete


@pytest.mark.parametrize(
    "text",
    [
        "{}",
        ' {"drafts": []}\n',
        '{"drafts": [{"instructions": "keep } [ and \\" quoted"}]}',
    ],
)
def test_authoring_json_root_boundary_accepts_one_complete_object(text: str) -> None:
    assert authoring_json_root_is_complete(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "[]",
        '{"drafts": [}',
        '{"drafts": []} commentary',
        'prefix {"drafts": []}',
        '{"drafts": []',
    ],
)
def test_authoring_json_root_boundary_never_extracts_or_repairs(text: str) -> None:
    assert not authoring_json_root_is_complete(text)


def test_authoring_json_root_boundary_requires_text() -> None:
    with pytest.raises(TypeError):
        authoring_json_root_is_complete(object())  # type: ignore[arg-type]
