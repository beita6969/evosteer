"""The production constructor binds explicit policies before any model call."""

import asyncio
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from skillev_private.evaluation import integrity_runtime

from skillev.evaluation.actor_sandbox import ActorSandbox
from skillev.evaluation.input_metric_contracts import IID_BENCHMARKS
from skillev.evaluation.integrity_results import require_paired_controls
from skillev.evaluation.step0_integrity import InferenceArm
from tests.evaluation.test_integrity_trained_policies import checkpoint_fixture, policy_arms
from tests.evaluation.test_step0_architecture import _profile
from tests.evaluation.test_step0_integrity_pipeline import panel
from tests.evaluation.test_step0_integrity_runtime import CleanTokenizer


class ServiceTokenizer(CleanTokenizer):
    chat_template = "synthetic-template"

    @property
    def inner(self):
        return self

    def __len__(self):
        return 128


def production_runtime(tmp_path, monkeypatch, *, config_overrides=None, source_panel=None):
    settings, observed = checkpoint_fixture(tmp_path)
    observed[0]["model_info"]["tokenizer_path"] = "/synthetic/tokenizer"
    observed[0]["server_info"]["server_args"]["context_length"] = 20000
    observed[0]["models"]["data"].append(
        {"id": "synthetic-base", "root": "/synthetic/Qwen3.5-9B", "parent": None}
    )
    monkeypatch.setattr(integrity_runtime, "_server", lambda *args, **kwargs: deepcopy(observed[0]))
    monkeypatch.setattr(
        integrity_runtime, "PublicServiceTokenizer", lambda path: ServiceTokenizer()
    )
    monkeypatch.setattr(
        integrity_runtime,
        "_interpreter_sandbox",
        lambda *args, **kwargs: ActorSandbox.current(Path.cwd() / "src"),
    )
    arms = policy_arms()
    evalplus_source = tmp_path / "official-evalplus"
    (evalplus_source / "evalplus").mkdir(parents=True)
    (evalplus_source / "evalplus" / "config.py").write_text(
        "DEFAULT_MIN_TIME_LIMIT = 4.0\nDEFAULT_GT_TIME_LIMIT_FACTOR = 4.0\n"
    )
    config = {
        "endpoints": ["http://127.0.0.1:1"],
        "arms": [arm.to_value() for arm in arms],
        "trained_policies": {"synthetic-policy-16": settings},
        "transport_attempts": 1,
        "concurrency": 2,
        "request_timeout_seconds": 10,
        "episode_timeout_seconds": 30,
        "context_length": 20000,
        "implementation_revision": "synthetic-routing",
        "bubblewrap": "/usr/bin/bwrap",
        "scorers": {"mbpp-plus": {"python": sys.executable, "source_root": str(evalplus_source)}},
        "thinking_policy": {
            "policy_id": "synthetic-thinking",
            "thinking_by_benchmark": {name: name == "aime-2026" for name in IID_BENCHMARKS},
        },
        "budgets": {
            "aime-2026": {
                "total_model_calls": 4,
                "total_output_tokens": 2048,
                "calls_per_turn": 4,
            }
        },
        "decoding": {"aime-2026": asdict(_profile(max_tokens=512))},
    }
    config.update(config_overrides or {})
    source = SimpleNamespace(
        panel=source_panel or panel(), interactive={}, targets={}, provenance="synthetic"
    )
    instance = integrity_runtime.PrivateIntegrityRuntime(config, source, tmp_path / "run")
    return instance, observed, arms


def test_non_mbpp_constructor_does_not_require_an_unrequested_native_scorer(tmp_path, monkeypatch):
    def unexpected_profile(settings):
        pytest.fail("a non-MBPP panel must not initialize EvalPlus")

    monkeypatch.setattr(integrity_runtime, "resolve_mbpp_profile", unexpected_profile)
    instance, _, arms = production_runtime(tmp_path, monkeypatch, config_overrides={"scorers": {}})
    try:
        assert instance.controls(arms[0], panel().entries).parser[
            "effective_benchmark_configuration"
        ]["aime-2026"]["native_thinking"]
        assert instance.mbpp_sandbox is instance.sandbox
    finally:
        asyncio.run(instance.close())


def test_requested_mbpp_still_initializes_its_native_profile(tmp_path, monkeypatch):
    from dataclasses import replace

    from skillev.evaluation.input_metric_contracts import PublicTaskView

    source = replace(
        panel(),
        entries=(PublicTaskView.from_record("synthetic", "mbpp-plus", {"prompt": "def f():"}),),
    )
    instance, _, _ = production_runtime(tmp_path, monkeypatch, source_panel=source)
    try:
        profile = instance.config["scorers"]["mbpp-plus"]["profile"]
        assert profile["min_time_limit"] == 4.0
        assert profile["gt_time_limit_factor"] == 4.0
    finally:
        asyncio.run(instance.close())


def test_constructor_controls_route_freezing_and_volatile_model_card_fields(tmp_path, monkeypatch):
    instance, observed, (base, trained) = production_runtime(tmp_path, monkeypatch)
    try:
        before = instance.controls(trained, panel().entries)
        effective = before.parser["effective_benchmark_configuration"]["aime-2026"]
        assert effective["native_thinking"]
        assert effective["decoding"]["enable_thinking"]
        assert effective["budgets"]["total_model_calls"] == 4
        assert (
            require_paired_controls(instance.controls(base, panel().entries), before, base, trained)
            == "forward-policy"
        )
        assert instance.generators[0].adapter_name is None
        assert (
            instance.trained_generators[trained.policy_id][0].adapter_name == "synthetic-forward-16"
        )
        assert instance._resolved_arm(panel().entries[0], trained).native_thinking
        for card in observed[0]["models"]["data"]:
            card["created"] = 999999
        observed[0]["models"]["data"].reverse()
        asyncio.run(instance.refresh())
        assert instance.controls(trained, panel().entries) == before
        route = next(card for card in observed[0]["models"]["data"] if card["parent"])
        route["root"] = "/synthetic/rebound-forward-adapter"
        with pytest.raises(ValueError):
            asyncio.run(instance.refresh())
    finally:
        asyncio.run(instance.close())


def test_controls_record_observed_prefix_cache_page_size(tmp_path, monkeypatch):
    instance, observed, (base, _) = production_runtime(tmp_path, monkeypatch)
    try:
        observed[0]["server_info"]["server_args"]["page_size"] = 64
        asyncio.run(instance.refresh())
        controls = instance.controls(base, panel().entries)
        assert controls.service["actual_server_arguments"][0]["page_size"] == 64
    finally:
        asyncio.run(instance.close())


def test_request_seed_does_not_claim_a_disabled_server_sampler_is_seeded(tmp_path, monkeypatch):
    instance, observed, (base, trained) = production_runtime(tmp_path, monkeypatch)
    try:
        unknown = instance.controls(base, panel().entries)
        assert unknown.service["sampling_seed_authority_by_replica"] == ["unverified"]
        observed[0]["server_info"]["version"] = "0.5.9"
        arguments = observed[0]["server_info"]["server_args"]
        arguments.update(
            enable_deterministic_inference=False, sampling_backend="flashinfer", random_seed=73
        )
        asyncio.run(instance.refresh())
        old = instance.controls(base, panel().entries)
        assert old.service["sampling_seed_authority_by_replica"] == ["server-global-rng"]
        assert old.service["actual_server_arguments"][0]["version"] == "0.5.9"
        assert old.service["actual_server_arguments"][0]["random_seed"] == 73
        arguments.update(enable_deterministic_inference=True, sampling_backend="pytorch")
        asyncio.run(instance.refresh())
        new = instance.controls(trained, panel().entries)
        assert new.service["sampling_seed_authority_by_replica"] == ["request-seed-enabled"]
        assert old.sampling == new.sampling
        # Identical requested decoding cannot hide a changed actual sampler in
        # a purported policy-only Step-0 versus trained-checkpoint comparison.
        with pytest.raises(ValueError):
            require_paired_controls(old, new, base, trained)
    finally:
        asyncio.run(instance.close())


def test_unknown_trained_policy_is_not_resolved_to_the_available_base_pool(tmp_path, monkeypatch):
    instance, _, _ = production_runtime(tmp_path, monkeypatch)
    try:
        with pytest.raises(ValueError):
            asyncio.run(
                instance.generate(
                    panel().entries[0],
                    InferenceArm("unknown", optimizer_steps=16, policy_id="not-configured"),
                    "synthetic",
                )
            )
        assert (
            instance.journal.connection.execute("SELECT COUNT(*) FROM executions").fetchone()[0]
            == 0
        )
    finally:
        asyncio.run(instance.close())
