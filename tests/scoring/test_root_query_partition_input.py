from types import SimpleNamespace

from skillev.scoring.objective import _encoded_query


class _Tokenizer:
    def encode(self, text: str) -> list[int]:
        return list(text.encode())


def test_partition_head_encodes_the_task_root_query_not_a_controller_constant() -> None:
    first = SimpleNamespace(initial_context=SimpleNamespace(query="first public task"))
    second = SimpleNamespace(initial_context=SimpleNamespace(query="second public task"))

    assert _encoded_query(_Tokenizer(), first) != _encoded_query(_Tokenizer(), second)


def test_partition_query_is_independent_of_changing_source_history() -> None:
    record = SimpleNamespace(
        initial_context=SimpleNamespace(
            query="stable public root query",
            source_messages=("initial state",),
        )
    )
    before = _encoded_query(_Tokenizer(), record)
    record.initial_context.source_messages = ("later public state",)
    assert _encoded_query(_Tokenizer(), record) == before
