import pytest
import torch

from skillev.policy.action_logprobs import action_token_logprobs


@pytest.mark.parametrize("length", [1, 15, 16, 39])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_chunked_target_preserves_all_tokens_full_vocabulary_and_signed_gradients(length, dtype):
    torch.manual_seed(0)
    source = torch.randn(length, 127, dtype=dtype)
    ids = torch.randint(0, 127, (length,))
    upstream = torch.linspace(-2, 3, length)
    reference, candidate = source.clone().requires_grad_(), source.clone().requires_grad_()
    expected = action_token_logprobs(reference, ids)
    actual = action_token_logprobs(candidate, ids, implementation="chunked-target@1")
    expected.backward(upstream)
    actual.backward(upstream)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        candidate.grad.float(),
        reference.grad.float(),
        atol=0.008 if dtype == torch.bfloat16 else 1e-6,
        rtol=0.008 if dtype == torch.bfloat16 else 1e-6,
    )
    # This component tolerance is not permission to relax the existing Qwen
    # end-to-end F/B/Z, residual and AdamW qualification thresholds.
