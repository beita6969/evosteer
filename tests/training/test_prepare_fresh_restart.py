"""Preparation contract exercised with a real tiny CPU dual-LoRA backbone."""

from dataclasses import replace

import pytest
import torch
from skillev_private.experiments import prepare_fresh_restart as prepare
from skillev_private.experiments.bayesian_training_setup import _read_preparation

from skillev.policy import QwenBackboneConfig, QwenMultimodalBackboneConfig, QwenPolicyBackbone


@pytest.fixture
def config(training_backbone_config):
    return QwenMultimodalBackboneConfig.from_value(
        replace(training_backbone_config, torch_dtype="bfloat16").to_value()
    )


def test_fresh_prepare_saves_independent_mapping_and_original_reader_contract(
    config, monkeypatch, tmp_path
):
    built = []

    def tiny_builder(declaration):
        # Production uses the multimodal builder; this tiny causal fixture runs
        # the real BF16 + two PEFT adapters + Z without a 9B model or GPU.
        backbone = QwenPolicyBackbone(QwenBackboneConfig.from_value(declaration.to_value()))
        built.append(backbone)
        return backbone

    def no_restore(*args, **kwargs):
        raise AssertionError("fresh preparation must never restore a checkpoint")

    monkeypatch.setattr(prepare, "build_qwen_policy_backbone", tiny_builder)
    monkeypatch.setattr(QwenPolicyBackbone, "load_checkpoint", no_restore)
    directory = tmp_path / "fresh"
    path = prepare.prepare_fresh_restart(backbone_config=config, output_root=directory)
    parsed, binding = _read_preparation(path)
    assert parsed == config
    assert binding.directory == str(directory / "initial-policy")
    mapping_path = directory / "initial_named_parameters.pt"
    mapping = torch.load(mapping_path, weights_only=True, map_location="cpu")
    current = built[0].named_trainable_parameters()
    assert set(mapping) == set(current)
    for name, tensor in mapping.items():
        assert tensor.device.type == "cpu"
        assert torch.equal(tensor, current[name].detach().cpu())
    assert any("lora_A" in name for name in mapping)
    assert any("lora_B" in name for name in mapping)
    assert any(name.startswith("z_head.") for name in mapping)
    assert not (directory / "config.json").exists()
    original = mapping_path.read_bytes()
    with pytest.raises(FileExistsError):
        prepare.prepare_fresh_restart(backbone_config=config, output_root=directory)
    assert len(built) == 1
    assert mapping_path.read_bytes() == original


@pytest.mark.parametrize(
    "invalid", ["seed", "z-seed", "dtype", "old-adapter", "old-policy", "cuda-unset"]
)
def test_invalid_preparation_never_builds_or_overwrites(config, monkeypatch, tmp_path, invalid):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid preparation must fail before model creation")

    monkeypatch.setattr(prepare, "build_qwen_policy_backbone", forbidden)
    seed = 1 if invalid == "seed" else 0
    if invalid == "z-seed":
        config = replace(config, z_initialization=replace(config.z_initialization, initial_seed=1))
    if invalid == "dtype":
        config = replace(config, torch_dtype="float32")
    if invalid in {"old-adapter", "old-policy"}:
        base = tmp_path / "old"
        base.mkdir()
        (
            base / ("adapter_config.json" if invalid == "old-adapter" else "policy_state.json")
        ).write_text("{}")
        config = replace(config, base_model_path=str(base))
    if invalid == "cuda-unset":
        config = replace(config, device="cuda")
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    directory = tmp_path / "new"
    with pytest.raises(ValueError):
        prepare.prepare_fresh_restart(backbone_config=config, output_root=directory, seed=seed)
    assert not directory.exists()


def test_failed_builder_leaves_exclusive_directory_without_completed_preparation(
    config, monkeypatch, tmp_path
):
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic model load failure")

    monkeypatch.setattr(prepare, "build_qwen_policy_backbone", fail)
    directory = tmp_path / "incomplete"
    with pytest.raises(RuntimeError):
        prepare.prepare_fresh_restart(backbone_config=config, output_root=directory)
    assert directory.is_dir()
    assert not (directory / "preparation.json").exists()
    with pytest.raises(FileExistsError):
        prepare.prepare_fresh_restart(backbone_config=config, output_root=directory)


def test_actual_preparation_to_new_application_passes_entry_helper(config, monkeypatch, tmp_path):
    import json

    from skillev_private.experiments.fresh_restart import require_clean_initial_application

    from skillev.experiments import _evolution_preflight_seed
    from tests.application.test_full_vertical_loop import build_application_fixture

    causal = QwenBackboneConfig.from_value(config.to_value())
    monkeypatch.setattr(prepare, "build_qwen_policy_backbone", lambda _: QwenPolicyBackbone(causal))
    preparation = prepare.prepare_fresh_restart(
        backbone_config=config, output_root=tmp_path / "prepared"
    )
    _, binding = _read_preparation(preparation)
    backbone = QwenPolicyBackbone(causal)
    # The later application may load the newly created original preparation;
    # preparation creation itself never loads an old adapter.
    backbone.load_checkpoint(binding.directory)
    root = tmp_path / "fresh-application"
    fixture = build_application_fixture(root, causal, backbone=backbone)
    documents = tuple(fixture.application.library.state.documents.values())
    monkeypatch.setattr(
        _evolution_preflight_seed, "planned_seed_documents", lambda profile: documents
    )
    (root / "evidence/queue").mkdir(parents=True)
    require_clean_initial_application(
        fixture.application,
        preparation=preparation,
        checkpoint_directory=preparation.parent / "initial-policy",
        root=root,
    )
    report = json.loads((root / "fresh-application-start.json").read_text())
    assert report["state"] == "verified"
    assert report["checks"]["preparation_tensors_match"] is True
    assert report["checks"]["forward_default_lora_b_zero"] is True
    assert report["checks"]["ledger_fully_settled"] is True
    assert report["checks"]["no_active_gradient_stream"] is True
