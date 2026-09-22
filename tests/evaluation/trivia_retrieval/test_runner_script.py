import pytest

from scripts.run_qwen35_trivia_public_retrieval import _messages, _parse_action, _Task


def test_formal_trivia_prompt_never_contains_accepted_answers() -> None:
    task = _Task("task", "Who wrote the work?", ("private answer",))
    messages = _messages(task, [])
    assert "private answer" not in str(messages)
    assert _parse_action('{"action":"search","value":"work author"}') == (
        "search",
        "work author",
    )
    with pytest.raises(ValueError):
        _parse_action("private answer")
