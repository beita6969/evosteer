from skillev.evaluation.current_iid.protocol14.code_eval import (
    CodeExtractionStatus,
    classify_python_completion,
)


def test_code_extraction_accepts_one_complete_fence_and_required_symbol() -> None:
    result = classify_python_completion(
        final_text="```python\ndef add(a, b):\n    return a + b\n```",
        reasoning_text="not used",
        required_symbols=("add",),
    )
    assert result.status is CodeExtractionStatus.VALID
    assert result.source == "def add(a, b):\n    return a + b"


def test_code_extraction_never_salvages_reasoning_channel() -> None:
    result = classify_python_completion(
        final_text="",
        reasoning_text="def add(a, b): return a + b",
        required_symbols=("add",),
    )
    assert result.status is CodeExtractionStatus.REASONING_ONLY
    assert result.source is None


def test_code_extraction_classifies_incomplete_syntax_and_missing_symbol() -> None:
    incomplete = classify_python_completion(
        final_text="```python\ndef add(a, b):\n    return a + b",
        reasoning_text=None,
        required_symbols=("add",),
    )
    syntax = classify_python_completion(
        final_text="def add(:\n    pass",
        reasoning_text=None,
        required_symbols=("add",),
    )
    missing = classify_python_completion(
        final_text="def subtract(a, b):\n    return a - b",
        reasoning_text=None,
        required_symbols=("add",),
    )
    assert incomplete.status is CodeExtractionStatus.INCOMPLETE_FENCE
    assert syntax.status is CodeExtractionStatus.SYNTAX_ERROR
    assert missing.status is CodeExtractionStatus.MISSING_REQUIRED_SYMBOL
