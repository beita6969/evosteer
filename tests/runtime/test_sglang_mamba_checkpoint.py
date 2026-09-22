"""Checkpoint storage precision must not change the prefill output operand."""

from types import SimpleNamespace

import pytest
import torch

from skillev.policy import sglang_mamba_checkpoint as repair


@pytest.mark.parametrize("variable_length", [False, True])
def test_fp32_checkpoint_and_original_output_precision(monkeypatch, variable_length):
    def original(
        k,
        w,
        u,
        g=None,
        gk=None,
        initial_state=None,
        initial_state_indices=None,
        save_new_value=True,
        cu_seqlens=None,
        chunk_indices=None,
    ):
        raise AssertionError("allocation launcher should be replaced, not rerun")

    class Kernel:
        def __getitem__(self, grid):
            def launch(**values):
                assert grid({"BV": 64}) == (1, 4 if variable_length else 2)
                assert values["IS_VARLEN"] == variable_length
                # A value lost by the old BF16 intermediate, before copying to
                # the persistent FP32 pool. No recurrence is simulated here.
                values["h"].fill_(1.001)

            return launch

    seen = []

    def output(q, h):
        seen.append(h)
        return h

    state = SimpleNamespace(
        CHUNK_SIZE=64,
        prepare_chunk_indices=lambda *args: (0, 1, 2),
        prepare_chunk_offsets=lambda *args: (0, 2, 3),
        chunk_gated_delta_rule_fwd_kernel_h_blockdim64=Kernel(),
    )
    chunk = SimpleNamespace(chunk_gated_delta_rule_fwd_h=original, chunk_fwd_o=output)
    modules = {
        "sglang.srt.layers.attention.fla.chunk_delta_h": state,
        "sglang.srt.layers.attention.fla.chunk": chunk,
    }
    monkeypatch.setattr(repair.importlib, "import_module", modules.__getitem__)
    repair.install_fp32_mamba_checkpoints()
    installed = chunk.chunk_gated_delta_rule_fwd_h
    repair.install_fp32_mamba_checkpoints()
    assert chunk.chunk_gated_delta_rule_fwd_h is installed
    k = torch.zeros((1, 130, 2, 4), dtype=torch.bfloat16)
    pool = torch.zeros((2, 2, 4, 4), dtype=torch.float32)
    h, values = installed(
        k,
        k,
        k,
        initial_state=pool,
        cu_seqlens=torch.tensor([0, 65, 130]) if variable_length else None,
    )
    assert h.dtype == pool.dtype
    assert h.shape == (1, 3, 2, 4, 4)
    assert values.dtype == k.dtype
    assert not torch.equal(h, h.to(k.dtype).float())
    result = chunk.chunk_fwd_o(k, h)
    torch.testing.assert_close(result, h.to(k.dtype), rtol=0, atol=0)
    assert seen[0].dtype == k.dtype
    assert h.dtype == torch.float32  # The checkpoint was not changed in place.
