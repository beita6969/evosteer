import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
from skillev_private.benchmarks.mbpp_scoring import MBPPScorerProfile
from skillev_private.benchmarks.protocol_v13_training_sessions import native_scorer_contracts
from skillev_private.evaluation.iid_architecture import (
    IIDSnapshotSelection,
    expanded_controls,
    freeze_iid_architecture,
    require_iid_architecture_match,
)
from skillev_private.evaluation.iid_episode_sources import (
    IIDSourceIdentity,
    canonical_source_key,
    public_episode_task,
    require_source_isolation,
)
from skillev_private.evaluation.integrity_sources import SourcePanel
from skillev_private.experiments.fresh_restart import load_fresh_config

from skillev.evaluation.healthbench_luna_profile import healthbench_condition
from skillev.evaluation.input_metric_contracts import IID_BENCHMARKS, PublicTaskView
from skillev.evaluation.integrity_pipeline import FrozenPanel
from skillev.runtime import SkillLibraryState
from skillev.training.rollout_workflow import RolloutWorkflowBinding


def synthetic_current_config():
    config = load_fresh_config(Path("configs/training/bayesianimprove_fresh_restart.yaml"))
    return replace(
        config,
        domains=IID_BENCHMARKS,
        thinking_off_domains=tuple(d for d in config.thinking_off_domains if d in IID_BENCHMARKS),
    )


def synthetic_native_panel(tmp_path: Path | None = None):
    game_directory = Path("/synthetic/json_2.1.1/eval/game")
    if tmp_path is not None:
        game_directory = tmp_path / "native-game"
        nested = game_directory / "nested"
        nested.mkdir(parents=True)
        (nested / "game.tw-pddl").write_text("synthetic game")
        (nested / "traj_data.json").write_text("{}")
    public = {
        "hotpotqa": {
            "question": "What is its color?",
            "context": [[f"Public passage {i}", ["It is blue."]] for i in range(10)],
        },
        "triviaqa": {"question": "Which color?", "public_context": "The public color is blue."},
        "aime-2026": {"problem": "Find the integer 2+3."},
        "healthbench": {"prompt": [{"role": "user", "content": "What is a regular bedtime?"}]},
        "alfworld": {"task": "Put a synthetic object in its destination."},
        "mbpp-plus": {"prompt": "def add_one(x):\n    # add one\n"},
        "humaneval": {"prompt": "def twice(x):\n    # multiply by two\n"},
    }
    entries = tuple(
        PublicTaskView.from_record(f"{domain}/0", domain, public[domain])
        for domain in IID_BENCHMARKS
    )
    identities = tuple(
        IIDSourceIdentity(
            e.task_id, e.benchmark, "original-0", "synthetic-panel", "synthetic-v1", "evaluation"
        )
        for e in entries
    )
    targets = {
        "aime-2026/0": {"answer": "5"},
        "mbpp-plus/0": {"private_target": {"secret": "not-public"}},
        "healthbench/0": {"prompt": public["healthbench"]["prompt"], "rubrics": ["hidden-rubric"]},
    }
    interactive = {
        "alfworld/0": {
            "case": {"deployment": "synthetic", "payload": {"game_id": "game"}},
            "manifest": {
                "deployments": {
                    "synthetic": {
                        "config_path": "/synthetic/alf-config.yaml",
                        "games": {
                            "game": {
                                "data_directory": str(game_directory),
                                "train_eval": "eval_out_of_distribution",
                            }
                        },
                    }
                }
            },
        }
    }
    panel = FrozenPanel(
        entries, "synthetic-test-only", "synthetic", tuple((d, 1) for d in IID_BENCHMARKS)
    )
    return SourcePanel(panel, targets, interactive, {"source": "synthetic"}), identities


def test_native_inputs_stay_native_and_source_aliases_cannot_hide_overlap():
    source, identities = synthetic_native_panel()
    tasks = [
        public_episode_task(e, i) for e, i in zip(source.panel.entries, identities, strict=True)
    ]
    assert "The public color is blue." in tasks[1].query
    assert tasks[3].model_visible_messages[0].role == "user"
    assert "hidden-rubric" not in json.dumps([t.to_value() for t in tasks])
    assert "not-public" not in json.dumps([t.to_value() for t in tasks])
    assert tasks[5].query == dict(source.panel.entries[5].fields)["prompt"]
    with pytest.raises(ValueError):
        require_source_isolation(
            source.panel.entries, identities, frozenset({("mbpp-plus", "original-0")})
        )


def test_freeze_two_readonly_arms_and_reject_non_snapshot_drift(make_training_harness, tmp_path):
    harness = make_training_harness()
    source, identities = synthetic_native_panel(tmp_path)
    config = synthetic_current_config()
    scorer = native_scorer_contracts(
        MBPPScorerProfile(), config.healthbench_judge, domains=config.domains
    )
    path = freeze_iid_architecture(
        destination=tmp_path / "frozen",
        architecture_id="synthetic-iid-architecture",
        source=source,
        identities=identities,
        excluded_sources={k: frozenset() for k in ("training", "development", "quality")},
        formal=config,
        initial_library=harness.library.state,
        model_controls={"model": "synthetic"},
        scorer_controls=scorer,
        environment_controls={"native_bridge": "synthetic"},
        serving_controls={"version": "synthetic"},
        workflow=RolloutWorkflowBinding(),
        acceptance_rules={d: {"metric": "synthetic", "minimum": 0.5} for d in IID_BENCHMARKS},
        prior_evaluation_sources=frozenset({("aime-2026", "original-0")}),
    )
    architecture = json.loads(path.read_text())
    assert architecture["arms"] == ["skills-off", "initial-library"]
    assert architecture["panel"]["prior_project_evaluation_overlap"] == [
        ["aime-2026", "original-0"]
    ]
    assert not architecture["panel"]["independent_unseen_final_test"]
    public = architecture["panel"]["public_inputs"]
    assert len(json.loads(public[0]["fields"]["context"])) == 10
    assert "hidden-rubric" not in json.dumps(public)
    assert "not-public" not in json.dumps(public)
    assert (
        architecture["controls"]["maximum_h0_tokens"]
        == config.application_config("test").maximum_h0_tokens
    )
    assert dict(config.sampling_config.reasoning_by_domain) == {
        domain: domain in {"aime-2026", "healthbench"} for domain in IID_BENCHMARKS
    }
    assert architecture["controls"]["sampling"] == json.loads(
        json.dumps(config.sampling_config.to_value())
    )
    reference = expanded_controls(
        architecture, IIDSnapshotSelection("initial-library", harness.generator.snapshot())
    )
    assert require_iid_architecture_match(reference, copy.deepcopy(reference)) == ()
    for group in (
        "formal",
        "serving",
        "scorers",
        "sampling",
        "environments",
        "domain_budgets",
        "actor_transport",
    ):
        changed = copy.deepcopy(reference)
        changed["controls"][group]["changed"] = True
        with pytest.raises(ValueError):
            require_iid_architecture_match(reference, changed)
    with pytest.raises(ValueError):
        IIDSnapshotSelection(
            "skills-off",
            harness.generator.snapshot(),
            library=SkillLibraryState.from_seed_documents(()),
        )
    off = expanded_controls(
        architecture, IIDSnapshotSelection("skills-off", harness.generator.snapshot())
    )
    assert not off["library_snapshot"]["active_skill_ids"]
    with pytest.raises(ValueError):
        require_iid_architecture_match(off, reference, allow_library_change=True)
    assert harness.loop.optimizer_step == harness.task_provider.cursor == 0


def test_old_implicit_default_cannot_start_fresh_architecture():
    with pytest.raises(ValueError):
        load_fresh_config(Path("configs/training/bayesianimprove_250.yaml"))


def test_native_healthbench_alias_does_not_hide_training_overlap():
    source, identities = synthetic_native_panel()
    identities = tuple(
        replace(row, source_question_id="public-uuid") if row.benchmark_id == "healthbench" else row
        for row in identities
    )
    aliases = {"healthbench": {"healthbench/public-uuid": "public-uuid"}}
    assert canonical_source_key(("healthbench", "healthbench/public-uuid"), aliases) == (
        "healthbench",
        "public-uuid",
    )
    with pytest.raises(ValueError):
        require_source_isolation(
            source.panel.entries,
            identities,
            frozenset({("healthbench", "healthbench/public-uuid")}),
            aliases,
        )


def test_explicit_six_domain_iid_has_no_humaneval_panel_or_score(make_training_harness, tmp_path):
    from dataclasses import replace

    config = load_fresh_config(
        Path("configs/training/bayesianimprove_autonomous_ttb_six_domain.yaml")
    )
    source, identities = synthetic_native_panel(tmp_path)
    source = replace(
        source,
        panel=FrozenPanel(
            tuple(e for e in source.panel.entries if e.benchmark in config.domains),
            "synthetic-test-only",
            "synthetic-six",
            tuple((d, 1) for d in config.domains),
        ),
    )
    identities = tuple(i for i in identities if i.benchmark_id in config.domains)
    scorers = {d: {"scorer": "synthetic"} for d in config.domains}
    scorers["healthbench"] = healthbench_condition(config.healthbench_judge)
    path = freeze_iid_architecture(
        destination=tmp_path / "six",
        architecture_id="six-domain-IID",
        source=source,
        identities=identities,
        excluded_sources={k: frozenset() for k in ("training", "development", "quality")},
        formal=config,
        initial_library=make_training_harness().library.state,
        model_controls={"model": "synthetic"},
        scorer_controls=scorers,
        environment_controls={"bridge": "synthetic"},
        serving_controls={"version": "synthetic"},
        workflow=RolloutWorkflowBinding(),
        acceptance_rules={d: {"metric": "synthetic", "minimum": 0.5} for d in config.domains},
    )
    value = json.loads(path.read_text())
    assert len(value["panel"]["records"]) == 6
    assert all(r["source"]["benchmark"] != "humaneval" for r in value["panel"]["records"])
    assert "humaneval" not in value["controls"]["scorers"]


def test_native_scorers_follow_declared_domains_without_changing_metrics():
    config = synthetic_current_config()
    mbpp = MBPPScorerProfile()
    current = native_scorer_contracts(mbpp, config.healthbench_judge, domains=config.domains)
    historical = native_scorer_contracts(
        mbpp, config.healthbench_judge, domains=(*config.domains, "humaneval")
    )
    assert "humaneval" not in current
    assert historical["humaneval"]["metric"] == "pass@1"
    for domain in config.domains:
        assert current[domain] == historical[domain]
    assert current["mbpp-plus"] == mbpp.to_value()
    assert current["healthbench"] == healthbench_condition(config.healthbench_judge)
