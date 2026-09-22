"""Synthetic source and transport fixtures; not actual model or benchmark evidence."""

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from skillev_private.benchmarks.evaluation_episode import (
    EvaluationEpisodeRecord,
    EvaluationSource,
    EvaluationTarget,
)
from skillev_private.evaluation.iid_architecture import record_value
from skillev_private.experiments import development_collection as dev

from skillev.evaluation.current_iid.protocol13.catalog import Protocol13Benchmark
from skillev.rollout import PolicySnapshot
from skillev.runtime import SkillLibraryState
from skillev.training.rollout_workflow import RolloutWorkflowBinding
from tests.training.fakes import make_public_tasks
from tests.v3_helpers import make_artifact


def write(path, value):
    path.write_text(json.dumps(value))
    return str(path)


def request_fixture(tmp_path, domain="healthbench"):
    task = replace(make_public_tasks(1)[0], task_id="synthetic-development-1", public_context={})
    record = EvaluationEpisodeRecord(
        EvaluationSource(
            Protocol13Benchmark(domain), "synthetic-pop", "synthetic-source", task.task_id
        ),
        task,
        EvaluationTarget({"synthetic_target": "not a licensed answer"}),
    )
    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps(record_value(record)) + "\n")
    policy = replace_policy(make_artifact("policy").manifest.policy_snapshot)
    manifest = {
        "comparison_id": "fixed-synthetic-pair",
        "seed": 0,
        "ordered_sources": [
            {
                "benchmark": domain,
                "source_id": "synthetic-source",
                "population_id": "synthetic-pop",
                "task_id": task.task_id,
                "source_dataset": "synthetic-historical-source",
                "source_revision": "synthetic-revision1",
                "source_split": "development",
                "report_domain": dev.DOMAINS[domain],
            }
        ],
    }
    request = {
        "format": dev.FORMAT,
        "condition_id": "candidate-old",
        "comparison_id": manifest["comparison_id"],
        "source_records": str(records),
        "source_manifest": write(tmp_path / "manifest.json", manifest),
        "source_aliases": write(tmp_path / "aliases.json", {}),
        "arm": "skills-off",
        "chunk_size": 1,
        "step0_policy": write(tmp_path / "policy.json", policy.to_value()),
        "initial_library": write(
            tmp_path / "library.json", SkillLibraryState.from_seed_documents(()).to_value()
        ),
        "exclusions": {
            name: write(tmp_path / f"excluded-{name}.json", [])
            for name in ("a0", "training", "quality")
        },
    }
    return request, record, policy


def replace_policy(policy):
    return PolicySnapshot.create(
        backbone_id=policy.backbone_id,
        forward_adapter_version="adapter-free",
        tokenizer_id=policy.tokenizer_id,
        backend_id=policy.backend_id,
        initial_trainable_state_hash=policy.initial_trainable_state_hash,
    )


@pytest.mark.parametrize("purpose", ["a0", "training", "quality"])
@pytest.mark.parametrize("domain", ["healthbench", "alfworld"])
def test_alias_overlap_is_rejected_before_any_live_dependency(
    tmp_path, monkeypatch, purpose, domain
):
    request, _, _ = request_fixture(tmp_path, domain)
    write(tmp_path / "aliases.json", {domain: {"alternate-native-id": "synthetic-source"}})
    write(
        tmp_path / f"excluded-{purpose}.json",
        [
            {
                "benchmark": domain,
                "source_id": "alternate-native-id",
                "population_id": "different-alias-pop",
            }
        ],
    )
    monkeypatch.setattr(dev, "observe_service", lambda *_: pytest.fail("overlap reached HTTP"))
    with pytest.raises(ValueError):
        asyncio.run(dev.run(request, tmp_path / "must-not-exist"))
    assert not (tmp_path / "must-not-exist").exists()


def test_fixed_comparison_allows_candidate_config_not_source_change(tmp_path):
    request, _, _ = request_fixture(tmp_path, "aime-2026")
    records, frozen = dev.selection(request)
    request["comparison_reference"] = write(tmp_path / "reference.json", frozen)
    request["condition_id"] = "candidate-new"
    assert dev.selection(request)[0] == records
    raw = json.loads((tmp_path / "records.jsonl").read_text())
    raw["public_task"]["query"] = "changed synthetic question"
    (tmp_path / "records.jsonl").write_text(json.dumps(raw) + "\n")
    with pytest.raises(ValueError):
        dev.selection(request)


def test_historical_aime_never_claims_aime2026_and_requires_provenance(tmp_path):
    request, _, _ = request_fixture(tmp_path, "aime-2026")
    _, frozen = dev.selection(request)
    assert frozen["source_manifest"]["ordered_sources"][0]["report_domain"] == "aime-historical"
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["ordered_sources"][0]["source_dataset"] = "AIME2026"
    write(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError):
        dev.selection(request)
    manifest["ordered_sources"][0]["source_dataset"] = "historical-synthetic"
    manifest["ordered_sources"][0]["source_revision"] = ""
    write(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError):
        dev.selection(request)


def test_unknown_outcomes_remain_null_and_cannot_pass_a0(tmp_path):
    request, _, _ = request_fixture(tmp_path)
    _, frozen = dev.selection(request)
    report = dev.aggregate(tmp_path, frozen, elapsed_seconds=2.0)
    assert report["domains"]["healthbench"]["success_rate"] is None
    assert report["consumed_budget"] is None
    assert report.get("baseline_accepted") is not True
    assert not report["eligible_for_a0_acceptance"]
    assert report["source_provenance"] == frozen["source_manifest"]["ordered_sources"]


@pytest.mark.parametrize("practice", [None, "training-development-skill-practice@1"])
@pytest.mark.parametrize("initialization_kind", [None, "fresh-forward", "skill-use-warmup"])
def test_live_wiring_uses_shared_collector_base_only_and_same_coordinates(
    tmp_path,
    monkeypatch,
    make_training_harness,
    training_backbone_config,
    practice,
    initialization_kind,
):
    request, record, policy = request_fixture(tmp_path)
    if practice is not None:
        request["development_practice"] = practice
    harness = make_training_harness()
    request.update(
        {
            "endpoint": "http://127.0.0.1:9",
            "mbpp_interpreter": "synthetic-python",
            "evalplus_source_root": "synthetic-official-source",
            "mbpp_profile": "synthetic-profile",
        }
    )
    for key in ("backbone", "formal_config", "deployments"):
        request[key] = write(tmp_path / key, {})
    profile = {"dtype": "bfloat16", "quantization": None, "served_model_name": "synthetic-base"}
    request["serving_profile"] = write(tmp_path / "profile.json", profile)
    backbone = SimpleNamespace(
        tokenizer_id=policy.tokenizer_id,
        base_model_path="synthetic-base-path",
        tokenizer_path=None,
        to_value=dict,
    )
    formal = SimpleNamespace(
        require_backbone=lambda _: None,
        sampling_config=replace(harness.config.rollout, base_seed=0),
        performance_profile="unused",
        to_value=lambda: {"explicit_candidate": "synthetic"},
        max_input_tokens=2048,
        maximum_reasoning_tokens=128,
        max_action_tokens=128,
        maximum_action_tokens=1024,  # An explicit larger domain cap must fit the service.
        healthbench_judge="synthetic-judge",
        domains=("hotpotqa",),
        hotpot_deliberation=False,
        task_budget=None,
        static_task_budget=None,
        domain_task_budgets={},
        application_config=lambda _: SimpleNamespace(trainer=harness.config),
    )
    gateway = None
    if initialization_kind is not None:
        from skillev.runtime.sglang_gateway import AdapterGeneration
        from tests.training.test_development_forward import initialization_inputs

        policy, initialized_config = initialization_inputs(
            tmp_path,
            request,
            harness.backbone,
            training_backbone_config,
            initialization_kind,
            formal.sampling_config.to_value(),
        )
        backbone.tokenizer_id = policy.tokenizer_id
        backbone.to_value = initialized_config.to_value
        gateway = SimpleNamespace(
            adapter_generation=AdapterGeneration(
                0, "synthetic-published-forward", policy.forward_adapter_version
            ),
            config=SimpleNamespace(supervisor_adapter="synthetic-published-forward"),
        )

        def bind(declaration, **kwargs):
            assert declaration["kind"] == initialization_kind
            assert kwargs["policy"] == policy
            return gateway

        monkeypatch.setattr(dev, "bind_forward_gateway", bind)
    monkeypatch.setattr(dev, "load_fresh_config", lambda _: formal)
    monkeypatch.setattr(
        dev, "QwenMultimodalBackboneConfig", SimpleNamespace(from_value=lambda _: backbone)
    )
    monkeypatch.setattr(
        dev,
        "QwenTokenizerAdapter",
        SimpleNamespace(from_config=lambda _: harness.generator.tokenizer),
    )
    monkeypatch.setattr(
        dev,
        "observe_service",
        lambda _: {"server_info": profile, "model_info": {"model_path": backbone.base_model_path}},
    )
    monkeypatch.setattr(dev, "serving_profile", lambda row: row)

    def require_service(*args, **kwargs):
        assert kwargs["minimum_context"] == formal.max_input_tokens + formal.maximum_action_tokens

    monkeypatch.setattr(dev, "require_training_service", require_service)
    monkeypatch.setattr(
        dev, "require_same_profile", lambda a, b: None if a == b else pytest.fail("profile drift")
    )
    monkeypatch.setattr(
        dev,
        "TrainingPerformanceConfig",
        SimpleNamespace(load=lambda _: SimpleNamespace(workflow=lambda: RolloutWorkflowBinding())),
    )
    monkeypatch.setattr(dev, "resolve_mbpp_profile", lambda _: object())
    monkeypatch.setattr(dev, "native_scorer_contracts", lambda *a, **kwargs: {"synthetic": True})
    calls = []

    async def sessions(rows, **kwargs):
        assert rows == (record,)
        calls.append("native-sessions")
        return (), object()

    monkeypatch.setattr(dev, "build_protocol13_training_sessions", sessions)

    def generator(**kwargs):
        assert kwargs["gateway"] is gateway
        assert kwargs["snapshot_provider"]() == policy
        return SimpleNamespace(close=lambda: calls.append("closed"))

    monkeypatch.setattr(dev, "ExternalSGLangRolloutGenerator", generator)

    async def collected(**kwargs):
        calls.append("shared-collector")
        assert kwargs["sampling_schedule_id"] == request["comparison_id"]
        assert kwargs["ordered_task_sequence_id"] == request["comparison_id"]
        assert kwargs["tasks"] == (record.input,)
        assert kwargs["sampled_policy_step"] == 0
        assert kwargs.get("quality_panel") is None
        assert isinstance(kwargs["base_sessions"], dev.PracticeSessionFactory) is (
            practice is not None
        )

    monkeypatch.setattr(dev, "collect_training_condition", collected)
    summary = asyncio.run(dev.run(request, tmp_path / "development-output"))
    assert calls == ["native-sessions", "shared-collector", "closed"]
    assert summary["record_kind"] == "development-evaluation"
    if initialization_kind is not None:
        assert summary["forward_initialization"]["kind"] == initialization_kind
        originals = tmp_path / "development-output" / "originals"
        assert (originals / "forward-preparation.json").read_bytes() == (
            tmp_path / "preparation.json"
        ).read_bytes()
        assert not (originals / "step0_policy").exists()
    assert summary["autonomous_skill_choice_measurement"] is (practice is None)
    assert (
        summary["training_updates"]
        == summary["posterior_updates"]
        == summary["skill_evolution"]
        == 0
    )
    with pytest.raises(FileExistsError):
        asyncio.run(dev.run(request, tmp_path / "development-output"))


def test_completed_native_values_are_not_changed_or_accepted_as_iid(tmp_path):
    from skillev_private.experiments.fresh_restart import require_iid_baselines

    request, _, _ = request_fixture(tmp_path, "healthbench")
    _, frozen = dev.selection(request)
    episodes = tmp_path / "collection" / "episodes"
    episodes.mkdir(parents=True)
    write(
        episodes / "episode-000000-private.json",
        {
            "artifact": {
                "record": {
                    "reward": {
                        "value": 0.7,
                        "success": False,
                        "native_metric_name": "synthetic-rubric",
                        "native_payload": {"public_metrics": {"synthetic-rubric": 0.41}},
                    }
                }
            }
        },
    )
    summary = dev.aggregate(tmp_path, frozen, elapsed_seconds=4.0)
    row = summary["domains"]["healthbench"]
    assert row["reward_mean"] == 0.7
    assert row["native_metrics"]["synthetic-rubric"] == 0.41
    assert row["success_count"] == 0
    assert row["completed"] == 1
    write(tmp_path / "summary.json", summary)
    write(tmp_path / "expanded-controls-private.json", {})
    references = tmp_path / "baseline-refs.json"
    write(references, {"skills-off": str(tmp_path), "initial-library": str(tmp_path)})
    with pytest.raises(ValueError):
        require_iid_baselines(references, config=None)


@pytest.mark.parametrize(
    ("source_split", "execution_mode", "valid"),
    [
        ("valid_seen", "valid_seen", False),
        ("valid_unseen", "valid_unseen", False),
        ("valid_seen", "eval_in_distribution", True),
        ("valid_unseen", "eval_out_of_distribution", True),
    ],
)
def test_alf_route_uses_official_mode_contract_before_service_or_reset(
    tmp_path, monkeypatch, source_split, execution_mode, valid
):
    request, _, _ = request_fixture(tmp_path, "alfworld")
    game = tmp_path / "one-compiled-synthetic-game"
    game.mkdir()
    (game / "game.tw-pddl").write_text("synthetic CPU fixture; never executed")
    (game / "traj_data.json").write_text("{}")
    raw = json.loads((tmp_path / "records.jsonl").read_text())
    raw["native_target"] = {
        "environment_route": {
            "game_file": str(game / "game.tw-pddl"),
            "mode": execution_mode,
        }
    }
    (tmp_path / "records.jsonl").write_text(json.dumps(raw) + "\n")
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["ordered_sources"][0].update(
        source_split=source_split, selection_stratum="synthetic-type"
    )
    write(tmp_path / "manifest.json", manifest)
    monkeypatch.setattr(
        dev, "observe_service", lambda *_: pytest.fail("invalid route reached HTTP")
    )
    from skillev_private.benchmarks.official_process import OfficialALFWorldProcessFactory

    monkeypatch.setattr(
        OfficialALFWorldProcessFactory,
        "create",
        lambda *_: pytest.fail("selection reset an environment"),
    )
    if valid:
        records, selected = dev.selection(request)
        assert records[0].output.target["environment_route"]["mode"] == execution_mode
        provenance = selected["source_manifest"]["ordered_sources"][0]
        assert provenance["source_split"] == source_split
        assert provenance["selection_stratum"] == "synthetic-type"
    else:
        with pytest.raises(ValueError):
            asyncio.run(dev.run(request, tmp_path / "must-not-dispatch"))
        assert not (tmp_path / "must-not-dispatch").exists()
