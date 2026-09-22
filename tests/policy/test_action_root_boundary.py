from __future__ import annotations

import pytest

from skillev.rollout.action_root_boundary import apply_action_json_root_boundary


class _CharacterTokenizer:
    def decode(self, token_ids: tuple[int, ...]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


@pytest.mark.parametrize(
    "text",
    [
        (
            '{"arguments":{"value":{"answer":"brace } in string"}},'
            '"kind":"complete","name":"complete"}'
        ),
        (
            '  {"arguments":{"items":[1,{"nested":true}]},"kind":"tool",'
            '"name":"x","resource_id":"r"}\n'
        ),
    ],
)
def test_action_boundary_accepts_one_complete_nested_object(text: str) -> None:
    tokens = tuple(ord(character) for character in text + "trailing prose")
    result = apply_action_json_root_boundary(_CharacterTokenizer(), tokens)

    assert result.matched is True
    assert _CharacterTokenizer().decode(result.content_token_ids) == text.rstrip()


@pytest.mark.parametrize(
    "text",
    [
        'prose {"kind":"complete"}',
        '```json\n{"kind":"complete"}\n```',
        '{"kind":"complete"} trailing-in-same-token',
        '{"kind":"complete"',
        '[{"kind":"complete"}]',
    ],
)
def test_action_boundary_does_not_extract_or_repair(text: str) -> None:
    # One token deliberately proves that non-whitespace trailing text in the
    # same sampled token cannot be silently sliced away.
    class _OneTokenTokenizer:
        def decode(self, token_ids: tuple[int, ...]) -> str:
            assert token_ids == (1,)
            return text

    result = apply_action_json_root_boundary(_OneTokenTokenizer(), (1,))
    assert result.matched is False
    assert result.content_token_ids == (1,)


@pytest.fixture
def fast_decoder():
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteDecoder
    from tokenizers.decoders import DecodeStream
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer

    backend = Tokenizer(BPE())
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
    backend.decoder = ByteDecoder()
    backend.train_from_iterator(
        ['{"code":"hello"}'], BpeTrainer(vocab_size=256, initial_alphabet=ByteLevel.alphabet())
    )

    class Decoder:
        calls = 0
        token_work = 0
        backend_tokenizer = backend

        def decode(self, ids, **kwargs):
            self.calls += 1
            self.token_work += len(ids)
            return backend.decode(list(ids), skip_special_tokens=False)

        def new_decode_stream(self):
            stream = DecodeStream(skip_special_tokens=False)
            return lambda token_id: stream.step(backend, token_id)

    return Decoder()


@pytest.mark.parametrize(
    "text",
    [
        ' {"unicode":"中文😀","code":"x = {1: \\"}\\"}","nested":[{},[]]} suffix',
        '{"str":"escaped \\" quote and \\\\ slash"}\nsecond',
        '{"incomplete": [1,2]',
        '{"mismatch": [1}}',
        'prose {"valid":true}',
        '{"special":"<|im_end|>","newline":"\\n"} suffix',
    ],
)
def test_fast_scanner_matches_general_prefix_oracle(fast_decoder, text):
    from skillev.rollout.action_root_boundary import _action_json_root_is_complete

    ids = tuple(fast_decoder.backend_tokenizer.encode(text).ids)
    expected = next(
        (
            ids[:i]
            for i in range(1, len(ids) + 1)
            if _action_json_root_is_complete(fast_decoder.decode(ids[:i]))
        ),
        None,
    )
    fast_decoder.calls = 0
    result = apply_action_json_root_boundary(fast_decoder, ids)
    assert result.matched == (expected is not None)
    assert result.content_token_ids == (ids if expected is None else expected)
    assert fast_decoder.calls <= 1


def test_long_action_scan_has_linear_decode_work(fast_decoder):
    import json

    text = json.dumps({"code": "x = {'brace': '}'}\n" * 2000})
    ids = tuple(fast_decoder.backend_tokenizer.encode(text + "unused tail").ids)
    result = apply_action_json_root_boundary(fast_decoder, ids)
    assert result.matched
    assert fast_decoder.calls == 1
    assert fast_decoder.token_work <= len(ids)


def test_server_post_sample_hook_stops_at_same_token_and_leaves_other_requests_alone(
    monkeypatch, fast_decoder
):
    from types import SimpleNamespace

    from skillev.runtime import sglang_action_boundary as hook

    class Request:
        def _check_str_based_finish(self, new_accepted_len=1):
            return False

    upstream = SimpleNamespace(Req=Request, FINISH_MATCHED_STR=lambda matched: matched)
    monkeypatch.setattr(hook.importlib, "import_module", lambda _: upstream)
    hook.install_action_root_boundary()
    req = Request()
    req.tokenizer = fast_decoder
    req.sampling_params = SimpleNamespace(
        custom_params={hook.ACTION_BOUNDARY_PARAMETER: hook.ACTION_JSON_ROOT_BOUNDARY_VERSION}
    )
    req.output_ids = []
    text = '{"code":"nested } braces","items":[{}]} unused tail'
    ids = fast_decoder.backend_tokenizer.encode(text).ids
    for token_id in ids:
        req.output_ids.append(token_id)
        if req._check_str_based_finish():
            break
    assert req.finished_len < len(ids)
    expected = apply_action_json_root_boundary(fast_decoder, tuple(ids))
    assert tuple(req.output_ids) == expected.content_token_ids
    req.sampling_params.custom_params = None
    assert req._check_str_based_finish() is False
