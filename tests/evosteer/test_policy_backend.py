import copy

import pytest

from skillev.contracts.canonical import canonical_json, stable_hash
from skillev.contracts.evosteer import DecisionRecord
from skillev.policy.action_trie import ActionTrie
from skillev.policy.evosteer import CausalLMOrchestrator, ContextWindowExceededError


@pytest.fixture
def policy():
    import torch
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    torch.manual_seed(42)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    vocab = {
        "<|endoftext|>": 0,
        "[UNK]": 1,
        **{
            character: i + 2
            for i, character in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))
        },
    }
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="<|endoftext|>",
        unk_token="[UNK]",
        pad_token="<|endoftext|>",
    )
    base = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_embd=16,
            n_layer=1,
            n_head=2,
            n_positions=4096,
            eos_token_id=0,
            bos_token_id=0,
            pad_token_id=0,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )
    actor = get_peft_model(
        base,
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["c_proj"],
            task_type="CAUSAL_LM",
            bias="none",
            fan_in_fan_out=True,
        ),
    )
    yield CausalLMOrchestrator(
        actor,
        tokenizer,
        reference_id="local-frozen@1",
        encoding_dim=16,
        # One token per byte: the orchestrator system prompt alone is ~900
        # tokens, so 1024 left the fixture prompts within a few bytes of the limit.
        context_window=2048,
        max_action_tokens=64,
    )
    torch.set_num_threads(previous_threads)


def menu_for(policy):
    return policy.menu(
        tuple(
            {"kind": kind, "node_id": node}
            for kind in ("RERUN_AGENT", "DROP_AGENT")
            for node in ("n0", "n1")
        )
    )


def copy_policy(policy, **overrides):
    return CausalLMOrchestrator(
        copy.deepcopy(policy.model),
        copy.deepcopy(policy.tokenizer),
        **{
            "reference_id": policy.reference_id,
            "encoding_dim": policy.encoding_dim,
            "context_window": policy.context_window,
            "max_action_tokens": policy.max_action_tokens,
            **overrides,
        },
    )


def record_for(policy, menu, path=None, prompt=None):
    path = path or menu.paths[0]
    state = {"history": [], "graph": {"nodes": ["n0", "n1"]}}
    return DecisionRecord(
        state_id=stable_hash(state),
        state_json=canonical_json(state),
        action_json=menu.actions[menu.paths.index(path)],
        prompt_ids=prompt or policy.encode_prompt("Repair the observed answer."),
        action_token_ids=path,
        legal_token_paths=menu.paths,
        features=(0.0,) * 30,
        reference_encoding=(),
        value_estimate=0.5,
    )


def test_masked_scores_sum_sequence_tokens_and_normalize_over_complete_actions(policy):
    import torch

    menu = menu_for(policy)
    probabilities = []
    for path in menu.paths:
        record = record_for(policy, menu, path)
        logits = policy._forward(policy.model, record.prompt_ids + path[:-1]).logits[0]
        masks = ActionTrie(menu.paths).masks(path)
        terms = [
            logits[len(record.prompt_ids) - 1 + i, list(allowed)]
            .float()
            .log_softmax(-1)[allowed.index(token)]
            for i, (token, allowed) in enumerate(zip(path, masks, strict=True))
        ]
        actual = policy.score(record)
        assert torch.allclose(actual, torch.stack(terms).sum(), atol=1e-7)
        assert not torch.allclose(actual, torch.stack(terms).mean(), atol=1e-4)
        probabilities.append(actual.exp().detach())
        assert torch.allclose(actual.detach(), policy.score(record, reference=True), atol=1e-7)
    assert torch.stack(probabilities).sum().item() == pytest.approx(1.0, abs=2e-7)


def test_combined_reference_pass_equals_separate_score_and_encoding(policy):
    import torch

    menu = menu_for(policy)
    for path in menu.paths:
        record = record_for(policy, menu, path)
        score, encoding = policy.reference_score_and_encoding(record)
        assert not score.requires_grad
        assert torch.allclose(score, policy.score(record, reference=True), atol=1e-6)
        assert encoding == pytest.approx(policy.encode_state(record.prompt_ids), abs=1e-5)
    single = policy.menu(({"kind": "STOP"},))
    record = record_for(policy, single)
    score, encoding = policy.reference_score_and_encoding(record)
    assert float(score) == 0.0
    assert encoding == policy.encode_state(record.prompt_ids)


def test_singleton_path_has_exact_zero_log_probability_and_zero_gradient(policy, monkeypatch):
    menu = policy.menu(({"kind": "STOP"},))
    record = record_for(policy, menu)

    def no_forward(*args, **kwargs):
        raise AssertionError("a singleton grammar must not execute the language model")

    monkeypatch.setattr(policy.model, "forward", no_forward)
    score = policy.score(record)
    assert score.item() == 0.0
    score.backward()
    assert all(p.grad is None or p.grad.count_nonzero() == 0 for p in policy.trainable_parameters())
    assert policy.sample(record.prompt_ids, menu, reference=False, seed=1) == menu.paths[0]


def test_cached_sampling_matches_full_prefill_with_same_random_stream(policy, monkeypatch):
    menu = menu_for(policy)
    prompt = policy.encode_prompt("Choose a legal repair.")
    original = policy._next_logits
    calls = []

    def tracked(model, ids, *, cache, cached_length):
        calls.append(
            (len(ids) - cached_length if cache is not None else len(ids), cache is not None)
        )
        return original(model, ids, cache=cache, cached_length=cached_length)

    monkeypatch.setattr(policy, "_next_logits", tracked)
    cached = [policy.sample(prompt, menu, reference=False, seed=seed) for seed in range(6)]
    assert any(used_cache and consumed < len(prompt) for consumed, used_cache in calls)

    def uncached(model, ids, *, cache, cached_length):
        return policy._forward(model, ids).logits[0, -1], None, 0

    monkeypatch.setattr(policy, "_next_logits", uncached)
    plain = [policy.sample(prompt, menu, reference=False, seed=seed) for seed in range(6)]
    assert cached == plain
    assert all(path in menu.paths for path in cached)


def test_actor_update_leaves_base_reference_and_state_encoding_frozen(policy):
    import torch

    menu = menu_for(policy)
    record = record_for(policy, menu)
    encoding = policy.encode_state(record.prompt_ids)
    before_reference = policy.score(record, reference=True).clone()
    before_base = {
        name: p.detach().clone()
        for name, p in policy.model.named_parameters()
        if "lora_" not in name
    }
    optimizer = torch.optim.SGD(policy.trainable_parameters(), lr=0.5)
    loss = -policy.score(record)
    assert loss.requires_grad
    assert not policy.score(record, reference=True).requires_grad
    loss.backward()
    assert any(
        p.grad is not None and p.grad.count_nonzero() > 0 for p in policy.trainable_parameters()
    )
    optimizer.step()
    assert policy.score(record).item() != pytest.approx(-loss.item(), abs=1e-6)
    assert torch.equal(before_reference, policy.score(record, reference=True))
    assert policy.encode_state(record.prompt_ids) == encoding
    assert len(encoding) == policy.encoding_dim
    for name, parameter in policy.model.named_parameters():
        if name in before_base:
            assert not parameter.requires_grad
            assert torch.equal(before_base[name], parameter)


def test_adapter_restore_is_exact_and_rejects_entire_bad_state_before_writing(policy):
    import torch

    saved = policy.adapter_state()
    assert saved
    assert all("lora_" in name for name in saved)
    menu = menu_for(policy)
    record = record_for(policy, menu)
    initial = policy.score(record).detach().clone()
    altered = {name: tensor + torch.randn_like(tensor) * 0.25 for name, tensor in saved.items()}
    policy.load_adapter_state(altered)
    assert not torch.equal(initial, policy.score(record).detach())
    invalid = {name: tensor.clone() for name, tensor in saved.items()}
    invalid[next(reversed(invalid))].fill_(float("nan"))
    before = policy.adapter_state()
    with pytest.raises(ValueError):
        policy.load_adapter_state(invalid)
    assert all(torch.equal(value, policy.adapter_state()[name]) for name, value in before.items())
    policy.load_adapter_state(saved)
    assert torch.equal(initial, policy.score(record).detach())


def test_explicit_debug_projection_survives_record_roundtrip_and_text_tampering_fails(policy):
    from dataclasses import replace

    policy = copy_policy(policy, context_mode="debug_head_tail")
    prompt = policy.encode_prompt("head " + "long public history " * 200 + " tail")
    assert len(prompt) == policy.context_window - policy.max_action_tokens
    record = record_for(policy, menu_for(policy), prompt=prompt)
    restored = DecisionRecord.from_value(record.to_value())
    assert restored.prompt_ids == prompt
    assert policy.score(restored).item() == policy.score(record).item()
    with pytest.raises(ValueError):
        policy.score(replace(restored, action_json=canonical_json({"kind": "STOP"})))


def test_full_default_preserves_middle_history_and_named_feature_vector(policy, monkeypatch):
    import json

    from skillev.orchestration.evosteer_features import FEATURE_NAMES, FEATURE_VERSION

    policy = copy_policy(policy, context_window=4096)
    payload = {
        "history": [
            {"event": "early", "output": "draft"},
            {"event": "unique-middle-history", "output": "critical correction"},
            {"event": "late", "output": "revised answer"},
        ],
        "features": [index / 30 for index in range(30)],
        "feature_schema": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
    }
    text = canonical_json(payload)
    prompt = policy.encode_prompt(text)
    assert policy.context_mode == "full"
    decoded = policy.tokenizer.decode(prompt, skip_special_tokens=False)
    messages = json.loads(decoded)
    assert messages[1]["content"] == text
    assert json.loads(messages[1]["content"]) == payload
    assert (
        tuple(policy.tokenizer.encode(canonical_json(messages), add_special_tokens=False)) == prompt
    )
    record = record_for(policy, menu_for(policy), prompt=prompt)
    seen = []
    original = policy._forward

    def forward(model, ids, **kwargs):
        seen.append(ids)
        return original(model, ids, **kwargs)

    monkeypatch.setattr(policy, "_forward", forward)
    actor_score = policy.score(record)
    reference_score = policy.score(record, reference=True)
    assert actor_score.item() == pytest.approx(reference_score.item())
    assert seen == [prompt + record.action_token_ids[:-1]] * 2


def test_full_overflow_is_explicit_and_debug_projection_is_exactly_reproducible(policy):
    # Longer than the 1024-token fixture window, yet the complete prompt, with
    # the orchestrator's action-semantics instruction, still fits in 4096.
    text = "early " + "middle history must not disappear " * 80 + " late"
    with pytest.raises(ContextWindowExceededError) as error:
        policy.encode_prompt(text)
    assert error.value.prompt_tokens > policy.context_window - policy.max_action_tokens
    assert error.value.reserved_tokens == policy.max_action_tokens
    assert error.value.context_window == policy.context_window
    complete = copy_policy(policy, context_window=4096)
    full_ids = complete.encode_prompt(text)
    debug = copy_policy(policy, context_mode="debug_head_tail")
    limit = policy.context_window - policy.max_action_tokens
    head = max(16, limit // 4)
    assert debug.encode_prompt(text) == full_ids[:head] + full_ids[-(limit - head) :]
    assert debug.configuration_id != policy.configuration_id


def test_exact_menu_reservation_keeps_full_prefix_and_sampling_scoring_support_equal(policy):
    text = "A complete public task and feedback history."
    prompt = policy.encode_prompt(text)
    menu = policy.menu(({"kind": "STOP"}, {"kind": "SET_OUTPUT", "node_id": "n0"}))
    reserve = max(map(len, menu.paths))
    narrow = copy_policy(policy, context_window=len(prompt) + reserve)
    with pytest.raises(ContextWindowExceededError):
        narrow.encode_prompt(text)
    assert narrow.encode_prompt(text, reserve_tokens=reserve) == prompt
    chosen = narrow.sample(prompt, menu, reference=False, seed=10)
    record = record_for(narrow, menu, path=chosen, prompt=prompt)
    assert narrow.score(record).isfinite()
    assert narrow.score(record, reference=True).isfinite()
    # A selected short path fitting the window is insufficient if another path
    # in the very same legal support does not fit the sampled action envelope.
    shortest = min(menu.paths, key=len)
    too_long = (2,) * (narrow.context_window - len(shortest))
    impossible = record_for(narrow, menu, path=shortest, prompt=too_long)
    with pytest.raises(ContextWindowExceededError):
        narrow.sample(too_long, menu, reference=False, seed=10)
    with pytest.raises(ContextWindowExceededError):
        narrow.score(impossible)
    with pytest.raises(ContextWindowExceededError):
        narrow.score(impossible, reference=True)


def test_context_configuration_respects_declared_model_and_tokenizer_limits(policy):
    assert policy.model_context_window == 4096
    with pytest.raises(ValueError, match="declared model/tokenizer limit"):
        copy_policy(policy, context_window=4097)
    tokenizer = copy.deepcopy(policy.tokenizer)
    tokenizer.model_max_length = 512
    with pytest.raises(ValueError, match="declared model/tokenizer limit"):
        CausalLMOrchestrator(
            copy.deepcopy(policy.model),
            tokenizer,
            reference_id="context-test",
            encoding_dim=16,
            context_window=1024,
            max_action_tokens=64,
        )
    for mode in ("head_tail", "silent", None, []):
        with pytest.raises(ValueError, match="context_mode"):
            copy_policy(policy, context_mode=mode)


def test_frozen_text_uses_cache_and_matches_uncached_generation(policy, monkeypatch):
    original = policy._next_logits
    cached_calls = []

    def tracked(model, ids, *, cache, cached_length):
        cached_calls.append(cache is not None)
        return original(model, ids, cache=cache, cached_length=cached_length)

    monkeypatch.setattr(policy, "_next_logits", tracked)
    cached = policy.frozen_text(
        "Answer this local task.", max_new_tokens=8, temperature=0.8, seed=7
    )
    assert any(cached_calls)

    def uncached(model, ids, *, cache, cached_length):
        return policy._forward(model, ids).logits[0, -1], None, 0

    monkeypatch.setattr(policy, "_next_logits", uncached)
    plain = policy.frozen_text("Answer this local task.", max_new_tokens=8, temperature=0.8, seed=7)
    assert cached == plain
    assert cached[1:] == (23, 8)


def test_reference_cannot_alias_trainable_actor_storage(policy):
    with pytest.raises(ValueError):
        CausalLMOrchestrator(
            policy.model,
            policy.tokenizer,
            reference_id="bad",
            encoding_dim=16,
            context_window=192,
            max_action_tokens=64,
            reference_model=policy.model,
        )
    independent = copy.deepcopy(policy.model)
    alternative = CausalLMOrchestrator(
        policy.model,
        policy.tokenizer,
        reference_id="independent",
        encoding_dim=16,
        context_window=policy.context_window,
        max_action_tokens=64,
        reference_model=independent,
    )
    assert all(not p.requires_grad for p in alternative.reference_model.parameters())


def test_local_huggingface_checkpoint_constructs_trainable_lora_backend(policy, tmp_path):
    from transformers import GPT2LMHeadModel

    base = GPT2LMHeadModel(copy.deepcopy(policy.model.config))
    base.save_pretrained(tmp_path)
    policy.tokenizer.save_pretrained(tmp_path)
    loaded = CausalLMOrchestrator.from_pretrained(
        str(tmp_path),
        reference_id="local-checkpoint@1",
        lora_rank=2,
        lora_alpha=4,
        target_modules=("c_proj",),
        context_window=policy.context_window,
        max_action_tokens=64,
    )
    assert loaded.encoding_dim == 16
    assert loaded.trainable_parameters()
    record = record_for(loaded, menu_for(loaded))
    assert loaded.score(record).requires_grad
    assert not loaded.score(record, reference=True).requires_grad


def test_configuration_identity_binds_tokenizer_adapter_architecture_and_model_pin(policy):
    def identity(*, tokenizer=None, model_pin=None, model=None):
        return CausalLMOrchestrator(
            model or copy.deepcopy(policy.model),
            tokenizer or copy.deepcopy(policy.tokenizer),
            reference_id=policy.reference_id,
            encoding_dim=16,
            context_window=policy.context_window,
            max_action_tokens=64,
            model_pin=model_pin,
        ).configuration_id

    assert identity() == policy.configuration_id
    tokenizer = copy.deepcopy(policy.tokenizer)
    tokenizer.chat_template = "{{ messages[0]['content'] }}"
    assert identity(tokenizer=tokenizer) != policy.configuration_id
    assert identity(model_pin="different-initial-artifact") != policy.configuration_id
    model = copy.deepcopy(policy.model)
    model.peft_config["default"].lora_alpha += 1
    assert identity(model=model) != policy.configuration_id


def test_trie_rejects_prefix_collisions_duplicates_mutability_and_boolean_tokens():
    for paths in (((1,), (1, 2)), ((1, 2), (1,)), ((1,), (1,)), ([1],), ((True,),)):
        with pytest.raises(ValueError):
            ActionTrie(paths)
    trie = ActionTrie(((1, 2), (1, 3, 2)))
    assert trie.allowed(()) == (1,)
    assert trie.allowed((1,)) == (2, 3)
    with pytest.raises(ValueError):
        trie.masks((1,))
