from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from training.backward_policy import _selected_token_logprob_sum_no_grad


def test_selected_logprob_projection_matches_full_logits() -> None:
    class Model:
        def __init__(self) -> None:
            self.positions = []

        def __call__(self, input_ids, *, attention_mask, logits_to_keep, use_cache):
            assert torch.equal(attention_mask, torch.ones_like(input_ids))
            assert use_cache is False
            self.positions.append(logits_to_keep.tolist())
            full = torch.nn.functional.one_hot(
                (input_ids + 1) % 7,
                num_classes=7,
            ).to(torch.float32)
            return SimpleNamespace(logits=full[:, logits_to_keep, :])

    model = Model()
    input_ids = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
    full_logits = torch.nn.functional.one_hot((input_ids + 1) % 7, num_classes=7).float()
    positions = torch.arange(1, 5)
    targets = input_ids[positions + 1]
    expected = (
        full_logits[positions].gather(1, targets.unsqueeze(1)).squeeze(1)
        - torch.logsumexp(full_logits[positions], dim=-1)
    ).sum()

    observed = _selected_token_logprob_sum_no_grad(
        model,
        input_ids,
        context_length=2,
        chunk_size=2,
    )

    assert observed == pytest.approx(float(expected.item()))
    assert model.positions == [[1, 2], [3, 4]]
