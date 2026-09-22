"""Skill-free warmup, the multi-family first author window and pooled author evidence.

A disclosed experimental schedule: no author window before author_start_batch,
then one window per family at once from the evidence pooled so far (which that
window consumes), then the paper's one-family-per-window lifecycle on each
window's own batch. Defaults reproduce the paper's schedule.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import shutil
from dataclasses import asdict, fields, replace

import pytest

from skillev.contracts.canonical import canonical_json, stable_hash
from skillev.evolution.evosteer_author import (
    FrozenSkillAuthor,
    SkillAuthorCallError,
    SkillAuthorOutputError,
    select_author_evidence,
)
from skillev.evolution.validated_admission import AdmissionLedger, SkillEntry
from skillev.evosteer_application import (
    AUTHOR_EVIDENCE_FORMAT,
    _IDENTITY_OPTIONAL_DEFAULTS,
    _IDENTITY_OPTIONAL_OPTIMIZER_DEFAULTS,
    EvoSteerConfig,
    _evidence_pool_from_value,
    _evidence_pool_value,
    _updated_evidence_pool,
)
from skillev.evosteer_cli import _application_config
from skillev.orchestration.graph import RoleSpec
from tests.evosteer.test_evosteer_author import Model, assessed, budget, procedure, trajectory
from tests.evosteer.test_v7_author import alternating_binding, family_binding

NEW_FIELDS = ("author_start_batch", "author_start_families", "author_evidence_pool")
# Also appended after runs were checkpointed, but declared beside the field each
# one qualifies rather than at the end.
INLINE_NEW_FIELDS = ("allow_repeated_pair_tasks", "validation_interval")
# The same, one level down: appended EvoOptimizerConfig fields.
NEW_OPTIMIZER_FIELDS = ("actor_beta1", "head_beta1")

# ---------------------------------------------------------------------------
# Configuration and CLI parsing (no torch required).
# ---------------------------------------------------------------------------


def config(**settings):
    return EvoSteerConfig(
        task_families=("alpha", "beta"), roles=(RoleSpec("solver", "Solve."),), **settings
    )


def test_new_fields_are_appended_and_defaults_keep_the_previous_identity():
    assert [item.name for item in fields(EvoSteerConfig)][-3:] == list(NEW_FIELDS)
    default = config()
    assert (default.author_start_batch, default.author_start_families) == (1, 1)
    assert default.author_evidence_pool == 0
    # Checkpoints and value-snapshot IDs of existing runs stay loadable.
    appended = (*NEW_FIELDS, *INLINE_NEW_FIELDS)
    legacy = {key: value for key, value in asdict(default).items() if key not in appended}
    legacy["optimizer"] = {
        key: value for key, value in legacy["optimizer"].items() if key not in NEW_OPTIMIZER_FIELDS
    }
    assert default.identity == str(stable_hash(legacy))
    for change in ({"author_start_batch": 9}, {"author_start_families": 2}):
        assert config(**change).identity != default.identity
    warm = config(author_start_batch=2)
    assert config(author_start_batch=2, author_evidence_pool=8).identity != warm.identity
    # A pool also works when the author interval alone delays the first window.
    assert config(author_interval=3, author_evidence_pool=2).first_author_batch == 3


@pytest.mark.parametrize(
    ("start", "interval", "expected"),
    [(1, 1, 1), (1, 3, 3), (10, 1, 10), (10, 4, 12), (8, 4, 8)],
)
def test_first_author_batch_is_the_first_scheduled_batch_from_the_start(start, interval, expected):
    assert (
        config(author_start_batch=start, author_interval=interval).first_author_batch == expected
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"author_start_batch": 0},
        {"author_start_batch": True},
        {"author_start_families": 0},
        {"author_start_families": 3},
        {"author_evidence_pool": -1},
        {"author_evidence_pool": 1},
        {"author_start_batch": 2, "author_evidence_pool": 2.0},
        {"author_start_batch": 2, "author_evidence_pool": False},
        # Without a batch before the first window the pool could never act.
        {"author_evidence_pool": 2},
        {"author_start_batch": 1, "author_interval": 1, "author_evidence_pool": 8},
    ],
)
def test_invalid_warmup_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        config(**settings)


def test_cli_parses_warmup_fields_and_accepts_a_disabled_pool():
    base = {
        "task_families": ["alpha", "beta"],
        "roles": [{"role_id": "solver", "instruction": "Solve."}],
    }
    parsed = _application_config(
        {**base, "author_start_batch": 9, "author_start_families": 2, "author_evidence_pool": 0}
    )
    assert (parsed.author_start_batch, parsed.author_start_families) == (9, 2)
    assert parsed.author_evidence_pool == 0
    pooled = {**base, "author_start_batch": 2, "author_evidence_pool": 12}
    assert _application_config(pooled).author_evidence_pool == 12
    assert _application_config({**base, "seed": 0}).seed == 0
    for bad in (
        {"author_start_batch": 0},
        {"author_start_families": 0},
        {"author_start_families": 3},
        {"author_evidence_pool": -1},
        {"author_evidence_pool": 1},
        {"author_evidence_pool": "8"},
        {"author_evidence_pool": 8},  # no warmup batch to collect from
    ):
        with pytest.raises(ValueError):
            _application_config({**base, **bad})


def test_validation_interval_round_trips_through_the_cli_and_keeps_the_default_identity():
    base = {
        "task_families": ["alpha", "beta"],
        "roles": [{"role_id": "solver", "instruction": "Solve."}],
    }
    assert _application_config(base).validation_interval == 20
    assert _application_config({**base, "validation_interval": 5}).validation_interval == 5
    for bad in ({"validation_interval": 0}, {"validation_interval": "5"}):
        with pytest.raises(ValueError):
            _application_config({**base, **bad})
    with pytest.raises(ValueError):
        config(validation_interval=0)
    # Checkpoints written before the field existed stay loadable: at its default
    # the field is left out of the identity they are bound to.
    assert _IDENTITY_OPTIONAL_DEFAULTS["validation_interval"] == config().validation_interval
    assert config(validation_interval=5).identity != config().identity


def test_optimizer_beta1_round_trips_through_the_cli_and_keeps_the_default_identity():
    base = {
        "task_families": ["alpha", "beta"],
        "roles": [{"role_id": "solver", "instruction": "Solve."}],
    }
    assert _application_config(base).optimizer.actor_beta1 == 0.9
    steered = _application_config({**base, "optimizer": {"actor_beta1": 0.95, "head_beta1": 0.0}})
    assert (steered.optimizer.actor_beta1, steered.optimizer.head_beta1) == (0.95, 0.0)
    for bad in ({"actor_beta1": 1.0}, {"head_beta1": -0.1}, {"actor_beta1": "0.95"}):
        with pytest.raises(ValueError):
            _application_config({**base, "optimizer": bad})
    # Checkpoints written before the fields existed stay loadable: at their
    # defaults they are left out of the identity those checkpoints are bound to.
    default = config()
    for name in NEW_OPTIMIZER_FIELDS:
        assert _IDENTITY_OPTIONAL_OPTIMIZER_DEFAULTS[name] == getattr(default.optimizer, name)
        assert config(optimizer=replace(default.optimizer, **{name: 0.95})).identity != (
            default.identity
        )


# ---------------------------------------------------------------------------
# Evidence pool helpers (no torch required).
# ---------------------------------------------------------------------------


def rollout(label, task_id, reward, *, batch, family="math", source="current"):
    """A risk-assessed rollout with the application's batch/sample identities."""
    batch_id = f"batch-{batch:06d}"
    base = trajectory(label, task_id, reward, family)
    values = {
        "sample_id": f"{batch_id}/{task_id}/{source}/{label}",
        "batch_id": batch_id,
        "source": source,
        "risk": None,
    }
    if source != "current":
        values["behavior_policy_id"] = base.reference_id
    if source.startswith("paired_"):
        values.update(
            pair_id=f"{batch_id}/{task_id}/pair",
            candidate_id="candidate",
            decisions=(replace(base.decisions[0], forced=True),),
        )
    return assessed(replace(base, **values))


def first_batch():
    return (
        rollout("mixed-ok", "mixed", 1.0, batch=1),
        rollout("easy-ok", "easy", 1.0, batch=1, source="natural_reference"),
        rollout("easy-ok-2", "easy", 0.9, batch=1),
        # Paired arms of the mixed task would form a contrast, but never enter.
        rollout("arm-ok", "mixed", 1.0, batch=1, source="paired_treatment"),
        rollout("arm-bad", "mixed", 0.0, batch=1, source="paired_control"),
        rollout("code-bad", "c", 0.0, batch=1, family="code"),
    )


def second_batch():
    return (
        rollout("mixed-bad", "mixed", 0.0, batch=2, source="natural_reference"),
        rollout("late-ok", "late", 1.0, batch=2),
    )


def test_pool_keeps_bounded_natural_rollouts_and_prefers_cross_batch_contrasts():
    families = ("math", "code")
    first = _updated_evidence_pool({}, first_batch(), families, 2)
    # No natural contrast yet: the best successes of distinct tasks (ties by ID).
    assert [item.sample_id.split("/")[-1] for item in first["math"]] == ["easy-ok", "mixed-ok"]
    assert [item.sample_id.split("/")[-1] for item in first["code"]] == ["code-bad"]
    second = _updated_evidence_pool(first, second_batch(), families, 2)
    # A success from batch 1 and a failure from batch 2 of the same task now
    # form the contrast the author ranks first; unpaired successes drop out.
    assert [item.sample_id.split("/")[-1] for item in second["math"]] == [
        "mixed-ok",
        "mixed-bad",
    ]
    assert second["code"] == first["code"]
    for pool in (first, second):
        items = [item for family in families for item in pool[family]]
        assert all(item.source in {"current", "natural_reference"} for item in items)
        assert all(len(pool[family]) <= 2 for family in families)
        assert len({item.sample_id for item in items}) == len(items)
    # Exactly the author's own rule over the old pool plus the new natural rollouts.
    assert second["math"] == select_author_evidence(
        (*first["math"], *second_batch()), "math", 2
    )


def saved(pool, families, capacity):
    # The checkpoint path: canonical JSON text, then parsed back.
    return json.loads(canonical_json(_evidence_pool_value(pool, families, capacity)))


def test_pool_value_roundtrips_exactly_and_rejects_invalid_records():
    families = ("math", "code")
    pool = _updated_evidence_pool(
        _updated_evidence_pool({}, first_batch(), families, 2), second_batch(), families, 2
    )
    value = saved(pool, families, 2)
    assert value["format"] == AUTHOR_EVIDENCE_FORMAT
    restored = _evidence_pool_from_value(
        value, families=families, capacity=2, batch_index=2, retired=False
    )
    assert restored == pool

    def rejects(mutate, *, capacity=2, batch_index=2, retired=False, match=None):
        broken = copy.deepcopy(value)
        mutate(broken)
        with pytest.raises(ValueError, match=match):
            _evidence_pool_from_value(
                broken,
                families=families,
                capacity=capacity,
                batch_index=batch_index,
                retired=retired,
            )

    rejects(lambda v: v.update(format="evosteer-author-evidence-pool@0"), match="format")
    rejects(lambda v: v.update(capacity=3), match="capacity")
    rejects(lambda v: v["families"].pop("code"), match="families")
    rejects(lambda v: v["families"].update(other=[]), match="families")
    rejects(lambda v: v["families"]["code"].extend(v["families"]["math"]), match="capacity")
    rejects(lambda v: v["families"]["code"].append(v["families"]["math"][0]), match="family")
    # Records must come from committed batches.
    rejects(lambda v: None, batch_index=1, match="committed")
    paired = rollout("arm", "mixed", 1.0, batch=1, source="paired_treatment").to_value()
    rejects(lambda v: v["families"]["math"].__setitem__(0, paired), match="natural")
    # A rejected risk assessment is not author evidence.
    unsafe = assessed(
        replace(rollout("unsafe", "u", 1.0, batch=1), risk=None), accepted=False
    ).to_value()
    rejects(lambda v: v["families"]["math"].__setitem__(0, unsafe), match="accepted")
    # An edited record no longer matches its bound risk assessment.
    rejects(lambda v: v["families"]["math"][0].update(reward=0.25), match="risk")
    # Once the first open window is committed it has consumed the pool: any
    # record left would reach a later window as stale pre-skill evidence.
    rejects(lambda v: None, retired=True, match="outlived")
    empty = saved({family: () for family in families}, families, 2)
    assert _evidence_pool_from_value(
        empty, families=families, capacity=2, batch_index=5, retired=True
    ) == {family: () for family in families}
    wide = saved(pool, families, 3)
    wide["families"]["math"].append(wide["families"]["math"][0])
    with pytest.raises(ValueError, match="repeats"):
        _evidence_pool_from_value(
            wide, families=families, capacity=3, batch_index=2, retired=False
        )


def test_select_author_evidence_is_the_authors_contrast_first_rule():
    items = (
        trajectory("easy-1", "easy", 1.0),
        trajectory("a-ok", "a", 1.0),
        trajectory("a-bad", "a", 0.0),
        trajectory("b-ok", "b", 0.75),
        trajectory("b-bad", "b", 0.25),
    )
    chosen = select_author_evidence(items, "math", 4)
    assert [item.sample_id for item in chosen] == ["a-ok", "a-bad", "b-ok", "b-bad"]
    model = Model()
    author = FrozenSkillAuthor(model, budget=budget())
    author.propose(items, family="math", window_id="window", seed=0)
    assert author.reports[0].selected_sample_ids == tuple(item.sample_id for item in chosen)
    with pytest.raises(TypeError):
        select_author_evidence(list(items), "math", 4)
    with pytest.raises(ValueError):
        select_author_evidence(items, "math", 0)


# ---------------------------------------------------------------------------
# Application integration (torch).
# ---------------------------------------------------------------------------


class WindowAuthor:
    """Records every window; proposes a family-specific candidate or raises."""

    def __init__(self, failures=None):
        self.failures = failures or {}
        self.windows = []

    def propose(self, trajectories, *, family, window_id, seed):
        assert isinstance(trajectories, tuple)
        self.windows.append((family, window_id, tuple(t.sample_id for t in trajectories)))
        if family in self.failures:
            raise self.failures[family]
        body = f"Procedure for {family} from {window_id}."
        return SkillEntry(
            f"skill-{family}-{len(self.windows)}",
            family,
            "sha256:" + hashlib.sha256(body.encode()).hexdigest(),
            body=body,
        )


class FamilyModel(Model):
    """A frozen author model whose procedure names the requested family."""

    def frozen_text(self, text, **kwargs):
        self.calls.append((text, kwargs))
        family = json.loads(text.split("\n", 1)[1])["family"]
        return canonical_json({**procedure(), "name": f"Check for {family}"}), 101, 23


def warmup_application(families, *, author=None, max_candidates=1, **settings):
    from skillev.evosteer_application import EvoSteerApplication
    from tests.evosteer.test_application import application

    source = application(candidate=False)
    config = replace(
        source.config,
        task_families=families,
        max_candidates_per_family=max_candidates,
        **settings,
    )
    return EvoSteerApplication(source.policy, config, author=author)


def empty_menu_id():
    return AdmissionLedger().freeze_menu("probe", "any").menu_id


def resign(directory, state):
    """Rewrite a checkpoint state with a matching checksum (a deliberate tamper)."""
    payload = canonical_json(state)
    (directory / "state.json").write_text(payload + "\n", encoding="utf-8")
    (directory / "state.sha256").write_text(hashlib.sha256(payload.encode()).hexdigest() + "\n")


def test_warmup_opens_no_window_then_authors_every_family_then_one_per_window():
    author = WindowAuthor()
    app = warmup_application(
        ("alpha", "beta"),
        author=author,
        max_candidates=2,
        author_start_batch=3,
        author_start_families=2,
    )

    def batch(number):
        return (
            family_binding(f"a{number}", "alpha", 0.0),
            family_binding(f"b{number}", "beta", 1.0),
        )

    for number in (1, 2):
        result = asyncio.run(app.train_batch(batch(number)))
        assert result.metrics["author"] == {"status": "warmup", "opens_at_batch": 3}
        assert author.windows == []
        assert not app.admission.skills
        assert all(item.menu_id == empty_menu_id() for item in result.trajectories)
        assert set(result.metrics["source_counts"]) == {"current", "natural_reference"}
    third = asyncio.run(app.train_batch(batch(3)))
    # The failing family ranks first and keeps the batch ID as its window.
    assert [(family, window) for family, window, _ in author.windows] == [
        ("alpha", "batch-000003"),
        ("beta", "batch-000003/beta"),
    ]
    report = third.metrics["author"]
    assert report["status"] == "candidate_proposed"
    assert (report["family"], report["window"]) == ("alpha", "batch-000003")
    assert [window["window"] for window in report["windows"]] == [
        "batch-000003",
        "batch-000003/beta",
    ]
    assert all(window["status"] == "candidate_proposed" for window in report["windows"])
    assert report["skill_id"] == report["windows"][0]["skill_id"]
    assert {skill.family for skill in app.admission.skills} == {"alpha", "beta"}
    # The authoring batch itself still ran without any skill.
    assert all(item.menu_id == empty_menu_id() for item in third.trajectories)
    fourth = asyncio.run(app.train_batch(batch(4)))
    assert "paired_treatment" in fourth.metrics["source_counts"]
    assert len(author.windows) == 3
    assert author.windows[-1][1] == "batch-000004"
    assert "windows" not in fourth.metrics["author"]


def test_first_window_is_capped_by_author_start_families_in_priority_order():
    author = WindowAuthor()
    app = warmup_application(
        ("alpha", "beta", "gamma"), author=author, author_start_families=2
    )
    bindings = (
        family_binding("a", "alpha", 1.0),
        family_binding("b", "beta", 0.0),
        family_binding("c", "gamma", 0.0),
    )
    prepared = app.prepare_batch(asyncio.run(app.collect_batch(bindings, batch_number=1)))
    assert [(family, window) for family, window, _ in author.windows] == [
        ("beta", "batch-000001"),
        ("gamma", "batch-000001/gamma"),
    ]
    assert len(prepared.author_result["windows"]) == 2
    assert sorted(skill.family for skill in app.admission.skills) == ["beta", "gamma"]


def test_an_author_error_in_one_family_does_not_block_the_others():
    author = WindowAuthor(
        failures={
            "alpha": SkillAuthorCallError("backend unavailable"),
            "beta": SkillAuthorOutputError("batch-000001/beta", "invalid JSON", "{"),
        }
    )
    app = warmup_application(
        ("alpha", "beta", "gamma"), author=author, author_start_families=3
    )
    bindings = tuple(family_binding(f"t-{f}", f, 0.0) for f in ("alpha", "beta", "gamma"))
    prepared = app.prepare_batch(asyncio.run(app.collect_batch(bindings, batch_number=1)))
    windows = {window["family"]: window for window in prepared.author_result["windows"]}
    assert windows["alpha"]["status"] == windows["beta"]["status"] == "author_error"
    assert "SkillAuthorCallError" in windows["alpha"]["error"]
    assert "SkillAuthorOutputError" in windows["beta"]["error"]
    assert windows["gamma"]["status"] == "candidate_proposed"
    assert len({window["window"] for window in windows.values()}) == 3
    assert prepared.author_result["status"] == "author_error"  # the first window's keys
    assert [skill.family for skill in app.admission.skills] == ["gamma"]


def test_pool_feeds_only_the_first_window_and_later_windows_see_their_own_batch():
    author = WindowAuthor()
    app = warmup_application(
        ("alpha",),
        author=author,
        max_candidates=2,
        author_start_batch=2,
        author_evidence_pool=2,
    )
    first = asyncio.run(
        app.train_batch((alternating_binding("mix", "alpha"), family_binding("ok", "alpha", 1.0)))
    )
    assert first.metrics["author"]["status"] == "warmup"
    pooled = app.author_evidence["alpha"]
    # The same-task success/failure contrast outranks the unpaired success.
    assert [item.task.task_id for item in pooled] == ["mix", "mix"]
    assert sorted(item.reward for item in pooled) == [0.0, 1.0]
    assert all(item.source in {"current", "natural_reference"} for item in pooled)
    assert first.metrics["author_evidence_pool"] == {"alpha": 2}
    second = asyncio.run(app.train_batch((family_binding("ok-2", "alpha", 1.0),)))
    assert second.metrics["author"]["status"] == "candidate_proposed"
    # The first open window sees the warmup pool followed by its whole batch...
    assert author.windows[0][2] == tuple(
        item.sample_id for item in (*pooled, *second.trajectories)
    )
    # ...and consumes the pool when its batch commits.
    assert app.author_evidence == {"alpha": ()}
    assert second.metrics["author_evidence_pool"] == {"alpha": 0}
    third = asyncio.run(app.train_batch((family_binding("ok-3", "alpha", 1.0),)))
    assert "paired_treatment" in third.metrics["source_counts"]
    assert third.metrics["author"]["window"] == "batch-000003"
    # A later window sees exactly its own batch, as in the paper, so warmup
    # contrasts cannot crowd out rollouts made under the new candidates.
    assert author.windows[1][2] == tuple(item.sample_id for item in third.trajectories)
    assert app.author_evidence == {"alpha": ()}


def test_a_failed_batch_leaves_the_pool_unchanged():
    app = warmup_application(("alpha",), author_start_batch=3, author_evidence_pool=4)
    asyncio.run(app.train_batch((alternating_binding("mix", "alpha"),)))
    before = app.author_evidence
    assert before["alpha"]
    collected = asyncio.run(
        app.collect_batch((family_binding("fresh", "alpha", 0.0),), batch_number=2)
    )
    prepared = app.prepare_batch(collected)
    assert prepared.evidence_pool["alpha"] != before["alpha"]  # staged only

    def broken_update(*args, **kwargs):
        raise RuntimeError("synthetic update failure")

    app.trainer.update = broken_update
    with pytest.raises(RuntimeError, match="synthetic update failure"):
        app.finish_batch(prepared)
    assert app.author_evidence == before
    assert app.batch_index == 1


def test_pool_and_multi_window_author_state_survive_checkpoints_exactly(tmp_path):
    def make():
        return warmup_application(
            ("alpha", "beta"),
            author=FrozenSkillAuthor(FamilyModel(), budget=budget()),
            author_start_batch=2,
            author_start_families=2,
            author_evidence_pool=3,
        )

    def batch(number):
        return (
            family_binding(f"a{number}", "alpha", 0.0),
            alternating_binding(f"b{number}", "beta"),
        )

    empty = {"alpha": (), "beta": ()}
    warm_app = make()
    asyncio.run(warm_app.train_batch(batch(1)))
    collected = warm_app.author_evidence
    assert all(collected[family] for family in empty)
    warm = warm_app.save_checkpoint(tmp_path / "warm")
    warm_state = json.loads((warm / "state.json").read_text())
    assert warm_state["author_evidence"]["format"] == AUTHOR_EVIDENCE_FORMAT
    # The warmup pool survives a restore exactly and feeds the first window.
    app = make()
    app.load_checkpoint(warm)
    assert app.author_evidence == collected
    asyncio.run(app.train_batch(batch(2)))
    reports = app.author.reports
    assert len(reports) == 2 and all(report.status == "candidate" for report in reports)
    assert reports[0].window_id == "batch-000002"
    assert reports[1].window_id == f"batch-000002/{reports[1].family}"
    for report in reports:
        assert any(item.startswith("batch-000001/") for item in report.selected_sample_ids)
    assert app.author.state_dict()["budget"]["settled"]["model_calls"] == 2
    assert {skill.family for skill in app.admission.skills} == {"alpha", "beta"}
    assert app.author_evidence == empty  # consumed by the first window
    path = app.save_checkpoint(tmp_path / "pooled")
    state = json.loads((path / "state.json").read_text())
    assert state["author_evidence"]["families"] == {"alpha": [], "beta": []}
    restored = make()
    restored.load_checkpoint(path)
    assert restored.author_evidence == app.author_evidence == empty
    assert restored.author.state_dict() == app.author.state_dict()
    assert restored.admission.to_value() == app.admission.to_value()
    result = asyncio.run(restored.train_batch((family_binding("a3", "alpha", 1.0),)))
    assert result.metrics["author_evidence_pool"] == {"alpha": 0, "beta": 0}

    def rejected(source, name, families, match):
        # A deliberate tamper with a valid checksum, rejected before any mutation.
        tampered = tmp_path / name
        shutil.copytree(source, tampered)
        changed = json.loads((source / "state.json").read_text())
        changed["author_evidence"]["families"] = families
        resign(tampered, changed)
        fresh = make()
        with pytest.raises(ValueError, match=match):
            fresh.load_checkpoint(tampered)
        assert fresh.batch_index == 0 and fresh.author_evidence == empty

    saved_rows = warm_state["author_evidence"]["families"]
    # A warmup record moved to another family.
    rejected(warm, "moved", {**saved_rows, "beta": saved_rows["alpha"]}, "family")
    # Warmup records smuggled past the first window would reach later windows.
    rejected(path, "stale", saved_rows, "outlived")


def test_default_configuration_keeps_the_single_window_report_and_checkpoint(tmp_path):
    from tests.evosteer.test_application import binding, with_author

    app = with_author()
    result = asyncio.run(app.train_batch((binding(),)))
    assert result.metrics["author"] == {
        "status": "no_proposal",
        "family": "synthetic",
        "window": "batch-000001",
    }
    assert "author_evidence_pool" not in result.metrics
    assert app.author_evidence == {"synthetic": ()}
    path = app.save_checkpoint(tmp_path / "default")
    state = json.loads((path / "state.json").read_text())
    assert "author_evidence" not in state
    appended = (*NEW_FIELDS, *INLINE_NEW_FIELDS)
    legacy = {key: value for key, value in asdict(app.config).items() if key not in appended}
    legacy["optimizer"] = {
        key: value for key, value in legacy["optimizer"].items() if key not in NEW_OPTIMIZER_FIELDS
    }
    assert state["config"] == str(stable_hash(legacy))


def test_a_warmup_checkpoint_carrying_skills_is_rejected(tmp_path):
    from tests.evosteer.test_v7_author import skill

    app = warmup_application(("alpha",), author_start_batch=3)
    asyncio.run(app.train_batch((family_binding("a", "alpha", 0.0),)))
    path = app.save_checkpoint(tmp_path / "warm")
    state = json.loads((path / "state.json").read_text())
    ledger = AdmissionLedger.from_value(state["admission"])
    ledger.propose(skill("smuggled", "alpha"), author_window_id="outside-any-window")
    state["admission"] = ledger.to_value()
    resign(path, state)
    fresh = warmup_application(("alpha",), author_start_batch=3)
    with pytest.raises(ValueError, match="skill-free warmup"):
        fresh.load_checkpoint(path)
    assert fresh.batch_index == 0 and not fresh.admission.skills


def test_warmup_rejects_an_injected_skill_library():
    from skillev.evosteer_application import EvoSteerApplication
    from tests.evosteer.test_application import application

    source = application(candidate=True)
    with pytest.raises(ValueError, match="empty initial skill library"):
        EvoSteerApplication(
            source.policy,
            replace(source.config, author_start_batch=2),
            admission=source.admission,
        )
