"""Owner-authorized unqualified long-run admission, never an IID acceptance result."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest
from skillev_private.experiments import bayesian_improve_training as entry
from skillev_private.experiments.bayesian_training_config import BayesianFormalConfig
from skillev_private.experiments.training_observation import (
    require_observation_sources,
    require_training_sources,
    unqualified_training_condition,
)

from tests.training import test_formal_training_observation as observation

CONFIG = observation.CONFIG
selection = observation.selection


@pytest.fixture
def authorized(selection):
    selected, path = selection
    value = json.loads(path.read_text())
    value.update(format="owner-authorized-unqualified-training@1", authorization="owner-explicit")
    path.write_text(json.dumps(value))
    return selected, path


def test_full_original_schedule_sources_and_unchanged_short_count(authorized):
    records, sources = authorized
    selected = entry.seven_domain_training_trajectories(records, steps=250)
    assert len(selected) == 7000
    require_training_sources(sources, selected, expected_trajectories=7000)
    require_observation_sources(sources, records)
    with pytest.raises(ValueError):
        require_observation_sources(sources, selected)
    with pytest.raises(ValueError):
        require_training_sources(sources, selected[:-1], expected_trajectories=7000)
    value = json.loads(sources.read_text())
    first = selected[0].episode
    value["source_aliases"] = {first.benchmark.value: {"other-population": first.source_id}}
    value["excluded_sources"]["development"] = [[first.benchmark.value, "other-population"]]
    sources.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        require_training_sources(sources, selected, expected_trajectories=7000)


def test_new_unqualified_condition_and_explicit_unchanged_same_run_resume(tmp_path, authorized):
    _, path = authorized
    config = BayesianFormalConfig.load(CONFIG)
    root = tmp_path / "run"
    assert unqualified_training_condition(config, sources=None, root=root, resume=None) == {}
    condition = unqualified_training_condition(config, sources=path, root=root, resume=None)
    declaration = condition["unqualified_training"]
    assert declaration["qualification"] == "unqualified"
    assert declaration["cold_start_failed"] is True
    assert declaration["a0_admission"] == "not-claimed"
    assert declaration["optimizer_steps"] == 250
    assert declaration["canonical_training_allowlist"]
    assert config.steps == 250
    assert config.run_plan.phase_search_steps == 249
    root.mkdir()
    saved = root / "unqualified-training-condition.json"
    saved.write_text(json.dumps(condition))
    original = root / "unqualified-training-sources-private.json"
    original.write_bytes(path.read_bytes())
    before = original.read_bytes(), saved.read_bytes()
    resume = root / "checkpoints" / "complete-boundary"
    assert (
        unqualified_training_condition(config, sources=path, root=root, resume=resume) == condition
    )
    with pytest.raises(ValueError):
        unqualified_training_condition(config, sources=None, root=root, resume=resume)
    value = json.loads(path.read_text())
    value["excluded_sources"]["quality"].append(["healthbench", "another-held-out"])
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        unqualified_training_condition(config, sources=path, root=root, resume=resume)
    assert before == (original.read_bytes(), saved.read_bytes())


@pytest.mark.parametrize(
    "conflict", ["iid", "short", "method", "old-run", "steps", "cadence", "undeclared"]
)
def test_authorization_does_not_bypass_other_conditions(tmp_path, authorized, conflict):
    _, path = authorized
    config = BayesianFormalConfig.load(CONFIG)
    kwargs = {"sources": path, "root": tmp_path / "run", "resume": None}
    if conflict == "iid":
        kwargs["iid_baselines"] = Path("not-read-iid")
    elif conflict == "short":
        kwargs["observation_steps"] = 5
    elif conflict == "method":
        kwargs["continuation"] = True
    elif conflict == "old-run":
        kwargs["resume"] = Path("old-complete-snapshot")
    elif conflict == "steps":
        config = replace(config, steps=5)
    elif conflict == "cadence":
        config = replace(config, checkpoint_every=1)
    else:
        value = json.loads(path.read_text())
        value.pop("authorization")
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        unqualified_training_condition(config, **kwargs)


def test_explicit_proactive_hint_continuation_keeps_source_authorization(tmp_path, authorized):
    _, path = authorized
    before = BayesianFormalConfig.load(Path("configs/training/bayesianimprove_autonomous_ttb.yaml"))
    after = replace(before, skill_exposure="catalog-then-read@3")
    root = tmp_path / "run"
    condition = unqualified_training_condition(before, sources=path, root=root, resume=None)
    root.mkdir()
    (root / "formal-config.json").write_text(json.dumps(before.to_value()))
    (root / "unqualified-training-condition.json").write_text(json.dumps(condition))
    (root / "unqualified-training-sources-private.json").write_bytes(path.read_bytes())
    resume = root / "checkpoints/complete-boundary"
    kwargs = {"sources": path, "root": root, "resume": resume, "continuation": True}
    with pytest.raises(ValueError):
        unqualified_training_condition(after, **kwargs)
    assert (
        unqualified_training_condition(after, **kwargs, proactive_catalog_continuation=True)
        == condition
    )
    for changed in (replace(after, z_learning_rate=0.002), replace(after, max_turns=30)):
        with pytest.raises(ValueError):
            unqualified_training_condition(changed, **kwargs, proactive_catalog_continuation=True)
    assert json.loads((root / "formal-config.json").read_text()) == before.to_value()


def test_explicit_domain_removal_keeps_original_authorization(tmp_path, authorized):
    from types import SimpleNamespace

    from skillev_private.experiments.bayesian_condition_transition import save_condition_transition

    from skillev.evaluation.external_judge_policy import (
        DIRECT_HEALTHBENCH_PROFILE,
        GATEWAY_HEALTHBENCH_PROFILE,
    )

    before = BayesianFormalConfig.load(
        Path("configs/training/bayesianimprove_autonomous_ttb_proactive_skills.yaml")
    )
    after = BayesianFormalConfig.load(
        Path("configs/training/bayesianimprove_autonomous_ttb_six_domain.yaml")
    )
    before = replace(before, healthbench_judge=DIRECT_HEALTHBENCH_PROFILE)
    after = replace(after, healthbench_judge=DIRECT_HEALTHBENCH_PROFILE)
    _, path = authorized
    root = tmp_path / "run"
    condition = unqualified_training_condition(before, sources=path, root=root, resume=None)
    root.mkdir()
    (root / "formal-config.json").write_text(json.dumps(before.to_value()))
    (root / "unqualified-training-condition.json").write_text(json.dumps(condition))
    (root / "unqualified-training-sources-private.json").write_bytes(path.read_bytes())
    snapshot = root / "checkpoints/step-00000004"
    current = unqualified_training_condition(
        after,
        sources=path,
        root=root,
        resume=snapshot,
        continuation=True,
        domain_subset_continuation=True,
    )
    assert current["unqualified_training"]["batch_size"] == 24
    app = SimpleNamespace(
        training_loop=SimpleNamespace(optimizer_step=4),
        snapshot_identity=SimpleNamespace(sampling_schedule_algorithm="unchanged"),
        evolution_loop=SimpleNamespace(save_condition_boundary=lambda name: None),
    )
    save_condition_transition(
        app, root=root, source_snapshot=snapshot, source=before, target=after, domain_subset=True
    )
    assert (
        unqualified_training_condition(after, sources=path, root=root, resume=snapshot) == current
    )
    assert json.loads((root / "unqualified-training-condition.json").read_text()) == condition
    gateway = replace(after, healthbench_judge=GATEWAY_HEALTHBENCH_PROFILE)
    assert (
        unqualified_training_condition(
            gateway,
            sources=path,
            root=root,
            resume=snapshot,
            continuation=True,
            healthbench_judge_continuation=True,
        )
        == current
    )
    with pytest.raises(ValueError):
        unqualified_training_condition(
            replace(gateway, static_max_turns=2),
            sources=path,
            root=root,
            resume=snapshot,
            continuation=True,
            healthbench_judge_continuation=True,
        )
    assert json.loads((root / "unqualified-training-condition.json").read_text()) == condition


def test_explicit_judge_change_preserves_unqualified_sources_and_method(tmp_path, authorized):
    from skillev.evaluation.external_judge_policy import (
        DIRECT_HEALTHBENCH_PROFILE,
        GATEWAY_HEALTHBENCH_PROFILE,
    )

    _, path = authorized
    before = replace(
        BayesianFormalConfig.load(CONFIG), healthbench_judge=DIRECT_HEALTHBENCH_PROFILE
    )
    after = replace(before, healthbench_judge=GATEWAY_HEALTHBENCH_PROFILE)
    root = tmp_path / "run"
    condition = unqualified_training_condition(before, sources=path, root=root, resume=None)
    root.mkdir()
    (root / "formal-config.json").write_text(json.dumps(before.to_value()))
    saved = root / "unqualified-training-condition.json"
    saved.write_text(json.dumps(condition))
    (root / "unqualified-training-sources-private.json").write_bytes(path.read_bytes())
    kwargs = {
        "sources": path,
        "root": root,
        "resume": root / "checkpoints/complete-boundary",
        "continuation": True,
    }
    with pytest.raises(ValueError):
        unqualified_training_condition(after, **kwargs)
    assert (
        unqualified_training_condition(after, **kwargs, healthbench_judge_continuation=True)
        == condition
    )
    for changed in (replace(after, z_learning_rate=0.002), replace(after, max_turns=30)):
        with pytest.raises(ValueError):
            unqualified_training_condition(changed, **kwargs, healthbench_judge_continuation=True)
    with pytest.raises(ValueError):
        unqualified_training_condition(
            after,
            **kwargs,
            healthbench_judge_continuation=True,
            domain_subset_continuation=True,
        )
    assert json.loads(saved.read_text()) == condition
    assert json.loads((root / "formal-config.json").read_text()) == before.to_value()


@pytest.mark.parametrize("allow_judge", [False, True])
def test_coordinator_passes_judge_authorization_before_runtime_start(
    tmp_path, authorized, monkeypatch, allow_judge
):
    from skillev_private.experiments import training_observation

    from skillev.evaluation.external_judge_policy import (
        DIRECT_HEALTHBENCH_PROFILE,
        GATEWAY_HEALTHBENCH_PROFILE,
    )

    _, sources = authorized
    before = replace(
        BayesianFormalConfig.load(CONFIG), healthbench_judge=DIRECT_HEALTHBENCH_PROFILE
    )
    after = replace(before, healthbench_judge=GATEWAY_HEALTHBENCH_PROFILE)
    root = tmp_path / "run"
    condition = unqualified_training_condition(before, sources=sources, root=root, resume=None)
    root.mkdir()
    (root / "formal-config.json").write_text(json.dumps(before.to_value()))
    (root / "unqualified-training-condition.json").write_text(json.dumps(condition))
    (root / "unqualified-training-sources-private.json").write_bytes(sources.read_bytes())

    class AuthorizedBeforeRuntimeError(Exception):
        pass

    def validate(*args, **kwargs):
        assert kwargs["healthbench_judge_continuation"] is allow_judge
        assert unqualified_training_condition(*args, **kwargs) == condition
        raise AuthorizedBeforeRuntimeError

    monkeypatch.setattr(training_observation, "unqualified_training_condition", validate)
    with pytest.raises(AuthorizedBeforeRuntimeError):
        asyncio.run(
            entry.run_coordinator(
                config=after,
                bindings=None,
                profile=None,
                topology=None,
                root=root,
                resume=root / "checkpoints/complete-boundary",
                unqualified_training_sources=sources,
                allow_healthbench_judge=allow_judge,
            )
        )
