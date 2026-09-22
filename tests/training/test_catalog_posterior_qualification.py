"""CPU synthetic scoring fixtures, never evidence of actual Qwen execution."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts import qualify_catalog_posterior as run
from skillev.calibration import CalibrationConfig
from skillev.diagnostics import DiagnosticsConfig
from skillev.policy import AdapterRole
from tests.v3_helpers import CharacterTokenizer, make_artifact


class TinyScorer:
    def __init__(self):
        self.tokenizer = CharacterTokenizer()
        self.weight = torch.nn.Parameter(torch.tensor(0.2))
        self.calls = []

    def score(self, prefix_ids, action_ids, role):
        self.calls.append((prefix_ids, action_ids, role))
        sign = 1 if role is AdapterRole.FORWARD_POLICY else -1
        return self.weight.expand(len(action_ids)) * sign - 1

    def z_value(self, query_ids):
        return self.weight.square()


def test_actual_shared_objective_and_projection_restore_in_new_process(tmp_path):
    backbone = TinyScorer()
    artifact = make_artifact(
        "synthetic-only", skills_by_step=(("skill-alpha",), ()), reward_success=True
    )
    before = backbone.weight.detach().clone()
    source = run.score_source(
        backbone,
        artifact,
        tmp_path,
        temperature_beta=1.0,
        forward_version=artifact.manifest.policy_snapshot.forward_adapter_version,
        backward_version="synthetic-backward@0",
    )
    assert len(source.edge_records) == artifact.record.horizon
    assert len(backbone.calls) == 2 * artifact.record.horizon
    assert all(e.step_importance == pytest.approx(0.4) for e in source.edge_records)
    assert backbone.weight.grad is None
    assert torch.equal(backbone.weight, before)
    report = run.project(
        source,
        "skill-alpha",
        tmp_path,
        diagnostics=DiagnosticsConfig(),
        calibration=CalibrationConfig(),
    )
    assert report["actual_optimizer_steps"] == 0
    assert report["formal_posterior_events"] == 0
    assert report["development_new_skill_event_count"] == 1
    saved = run.read(tmp_path / "posterior-batch.json")
    assert saved["updates"][0]["alpha_before"] == 1
    assert saved["updates"][0]["beta_count_before"] == 1
    command = [
        sys.executable,
        "-m",
        "scripts.qualify_catalog_posterior",
        "restore",
        "--output",
        str(tmp_path),
    ]
    subprocess.run(command, check=True, timeout=30, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})  # noqa: S603
    restored = run.read(tmp_path / "restored.json")
    assert restored["driver_pid"] != os.getpid()
    assert restored["new_skill_cells"] == report["new_skill_cells"]
    assert restored["new_posterior_updates"] == 0
    assert restored["teacher_forced_calls"] == 0


def test_no_actual_skill_call_cannot_be_invented_as_posterior(tmp_path):
    artifact = make_artifact("no-read", skills_by_step=((),))
    source = run.score_source(
        TinyScorer(),
        artifact,
        tmp_path,
        temperature_beta=1.0,
        forward_version=artifact.manifest.policy_snapshot.forward_adapter_version,
        backward_version="synthetic-backward@0",
    )
    with pytest.raises(ValueError):
        run.project(
            source,
            "new-skill",
            tmp_path,
            diagnostics=DiagnosticsConfig(),
            calibration=CalibrationConfig(),
        )
    assert not (tmp_path / "projection-after.json").exists()


def test_repeated_actual_reads_keep_accumulating_after_the_first_prior(tmp_path):
    artifact = make_artifact("repeated-read", skills_by_step=(("skill-alpha",), ("skill-alpha",)))
    source = run.score_source(
        TinyScorer(),
        artifact,
        tmp_path,
        temperature_beta=1.0,
        forward_version=artifact.manifest.policy_snapshot.forward_adapter_version,
        backward_version="synthetic-backward@0",
    )
    result = run.project(
        source,
        "skill-alpha",
        tmp_path,
        diagnostics=DiagnosticsConfig(),
        calibration=CalibrationConfig(),
    )
    events = run.read(tmp_path / "posterior-batch.json")["updates"]
    assert result["development_new_skill_event_count"] == 2
    assert events[0]["alpha_after"] == events[1]["alpha_before"]
    assert events[0]["beta_count_after"] == events[1]["beta_count_before"]


def test_failure_label_is_preserved_not_promoted_to_success(tmp_path):
    artifact = make_artifact("failure", reward_success=False)
    source = run.score_source(
        TinyScorer(),
        artifact,
        tmp_path,
        temperature_beta=1.0,
        forward_version=artifact.manifest.policy_snapshot.forward_adapter_version,
        backward_version="synthetic-backward@0",
    )
    run.project(
        source,
        "skill-alpha",
        tmp_path,
        diagnostics=DiagnosticsConfig(),
        calibration=CalibrationConfig(),
    )
    update = run.read(tmp_path / "posterior-batch.json")["updates"][0]
    assert not update["outcome"]
    assert update["alpha_after"] == 1
    assert update["beta_count_after"] > 1


def test_initial_mapping_rejects_old_forward_adapter_and_mismatched_z():
    reference = {
        f"policy.lora_{part}.{role.value}.weight": torch.ones(2) if part == "A" else torch.zeros(2)
        for role in AdapterRole
        for part in ("A", "B")
    }
    reference["z_head.weight"] = torch.ones(2)

    class Named:
        def named_trainable_parameters(self):
            return reference

    run.require_initial_parameters(Named(), {k: v.clone() for k, v in reference.items()})
    independent = {k: v.clone() for k, v in reference.items()}
    reference["z_head.weight"][0] = 2
    with pytest.raises(ValueError):
        run.require_initial_parameters(Named(), independent)
    reference["policy.lora_B.forward-policy.weight"][0] = 1
    with pytest.raises(ValueError):
        run.require_initial_parameters(Named(), {k: v.clone() for k, v in reference.items()})


def test_score_attempt_is_exclusive_and_failure_retained(tmp_path):
    root, output = tmp_path / "development", tmp_path / "posterior"
    root.mkdir()
    with pytest.raises(ValueError):
        run.score_run(root, output, tmp_path / "missing.json", 1.0)
    assert json.loads((output / "failure.json").read_text())["error_type"] == "ValueError"
    with pytest.raises(FileExistsError):
        run.score_run(root, output, tmp_path / "missing.json", 1.0)
    with pytest.raises(ValueError):
        run.score_run(root, root / "nested", Path("unused"), 1.0)


def test_adapter_free_artifact_cannot_be_relabelled_as_fresh_scorer(tmp_path):
    artifact = make_artifact("original-policy")
    backbone = TinyScorer()
    with pytest.raises(ValueError):
        run.score_source(
            backbone,
            artifact,
            tmp_path,
            temperature_beta=1.0,
            forward_version="different-fresh-forward@0",
            backward_version="B@0",
        )
    assert not backbone.calls
    assert not list(tmp_path.iterdir())


def test_fresh_read_reuses_author_copy_without_touching_original(tmp_path):
    from scripts import qualify_catalog_authoring as author
    from scripts import qualify_catalog_fresh_read as fresh
    from tests.training.test_catalog_authoring_qualification import Transport, setup

    tokenizer = CharacterTokenizer()
    root = tmp_path / "accepted-author"
    plan, result = setup(root, tokenizer)
    plan["policy"] = make_artifact("identity").manifest.policy_snapshot.to_value()
    (root / "plan.json").write_text(json.dumps(plan))
    library, skill = author.recover_mutation(
        root, plan, tokenizer, transport=Transport(tokenizer, result)
    )
    originals = {p.name: p.read_bytes() for p in root.glob("*.json")}
    output = tmp_path / "fresh-read"
    output.mkdir()
    recovered, recovered_skill = fresh.recover_author_copy(root, output, tokenizer)
    assert recovered.state == library.state
    assert recovered_skill == skill
    assert fresh.read(output / "author-origin.json")["new_author_generations"] == 0
    assert all((root / name).read_bytes() == content for name, content in originals.items())
    assert not list((output / "author-recovery" / root.name).glob("author-http-*.json"))


@pytest.mark.parametrize("published", [True, False])
def test_fresh_generator_binds_only_existing_real_route(tmp_path, published):
    from scripts import qualify_catalog_fresh_read as fresh
    from skillev.runtime.sglang_gateway import (
        SGLangGateway,
        SGLangGatewayConfig,
        SGLangGatewayError,
    )

    calls = []

    class Control:
        def request(self, **kwargs):
            calls.append(kwargs)
            assert kwargs["method"] == "GET"
            if kwargs["url"].endswith("/health"):
                return 200, {}
            return 200, {
                "data": [
                    {"id": name} for name in (["base", "published-F0"] if published else ["base"])
                ]
            }

    policy = make_artifact("fresh-identity").manifest.policy_snapshot
    plan = {
        "endpoint": "http://127.0.0.1:9",
        "base_model": "base",
        "published_forward_adapter": "published-F0",
        "policy": policy.to_value(),
    }
    gateway = SGLangGateway(
        SGLangGatewayConfig(
            endpoint_base=plan["endpoint"],
            base_model="base",
            supervisor_adapter="published-F0",
            control_retries=0,
        ),
        _control_transport=Control(),
    )
    if published:
        generator, counted = fresh.fresh_generator(
            tmp_path, plan, CharacterTokenizer(), gateway=gateway
        )
        try:
            assert generator.gateway is gateway
            assert generator.snapshot() == policy
            assert gateway.adapter_generation.adapter_revision == policy.forward_adapter_version
            assert counted.calls == 0
        finally:
            generator.close()
    else:
        with pytest.raises(SGLangGatewayError):
            fresh.fresh_generator(tmp_path, plan, CharacterTokenizer(), gateway=gateway)
        assert not (tmp_path / "actor-requests.sqlite3").exists()
    assert len(calls) == 2
