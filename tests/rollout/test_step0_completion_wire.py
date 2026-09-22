from skillev.evaluation.legacy_step0_completion import (
    StepZeroTerminalMode,
    native_action_constraint,
    project_terminal_candidate,
    render_terminal_prefix,
    terminal_constraint,
)


def test_aime_accepts_official_leading_zero_representation() -> None:
    assert project_terminal_candidate(StepZeroTerminalMode.AIME_INTEGER, "042") == (r"\boxed{42}")
    assert project_terminal_candidate(StepZeroTerminalMode.AIME_INTEGER, "1000") is None
    assert terminal_constraint(StepZeroTerminalMode.AIME_INTEGER).regex == r"[0-9]{1,3}"


def test_python_source_round_trips_without_json_escaping() -> None:
    source = 'def answer(value: str) -> str:\n    return value + "!"\n'
    assert project_terminal_candidate(StepZeroTerminalMode.PYTHON_SOURCE, source) == source.strip()
    assert terminal_constraint(StepZeroTerminalMode.PYTHON_SOURCE).json_schema is None


def test_native_action_schema_has_no_memory_argument() -> None:
    constraint = native_action_constraint("alfworld")
    assert isinstance(constraint.json_schema, dict)
    arguments = constraint.json_schema["properties"]["arguments"]  # type: ignore[index]
    assert "command" in arguments["properties"]  # type: ignore[operator]
    assert "memory" not in arguments["properties"]  # type: ignore[operator]


def test_terminal_prefix_does_not_reintroduce_the_json_action_contract() -> None:
    prompt = render_terminal_prefix(
        "private-style public initial envelope that must not be copied\n",
        reasoning_text="The supported short answer is the named entity.",
        mode=StepZeroTerminalMode.SHORT_ANSWER,
        public_question="Which named entity is supported?",
    )

    assert "structured-action-json" not in prompt
    assert "exactly one complete raw" not in prompt
    assert "Which named entity is supported?" in prompt
    assert "private-style public initial envelope" not in prompt
    assert "Do not copy a sentence" in prompt
    assert prompt.endswith("Terminal payload:\n")


def test_aime_terminal_transcribes_reasoning_without_resolving_the_task() -> None:
    prompt = render_terminal_prefix(
        "public task that must not be reintroduced\n",
        reasoning_text=r"The verified conclusion is \boxed{42}.",
        mode=StepZeroTerminalMode.AIME_INTEGER,
    )

    assert "public task that must not be reintroduced" not in prompt
    assert r"\boxed{42}" in prompt
    assert "last explicitly boxed integer" in prompt
    assert "Do not solve the task again" in prompt
