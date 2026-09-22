import asyncio
import json
from dataclasses import asdict, replace

import pytest
from skillev_private.experiments.initial_quality_probe import import_initial_quality_probe
from skillev_private.experiments.quality_panel import FixedQualityPanel, PanelSlot

from skillev.training.quality_gate import ProtocolProbe, QualityGatePolicy, QualityRule
from skillev.training.quality_monitor import QualityCheckpointStop, QualityProbeCoordinator


@pytest.fixture
def fixture(tmp_path):
    panel = FixedQualityPanel(
        "panel", "condition", (PanelSlot("occurrence", "synthetic", "pool", "source"),)
    )
    policy = QualityGatePolicy(
        "t0-floor",
        "panel",
        "condition",
        2,
        1,
        2,
        (QualityRule("success", 0.25, 0.2, baseline_minimum=0.5),),
    )
    monitor = QualityCheckpointStop(tmp_path / "quality", policy)
    probe = ProtocolProbe("completed", "panel", "condition", "initial", 0, 1, {"success": 0.75})
    source = tmp_path / "original.json"
    source.write_text(json.dumps(asdict(probe), indent=4) + "\n\n")
    return source, monitor, panel, probe


def install(fixture, **kwargs):
    source, monitor, panel, _ = fixture
    return import_initial_quality_probe(
        source,
        monitor=monitor,
        panel=panel,
        **{"optimizer_step": 0, "policy_snapshot_id": "initial", **kwargs},
    )


def test_original_bytes_preserved_and_initial_check_never_resamples(fixture):
    source, monitor, _, probe = fixture
    assert install(fixture) == probe
    saved = monitor.root / "probe-00000000.json"
    assert saved.read_bytes() == source.read_bytes()
    note = monitor.root / "initial-probe-import.json"
    original_note = note.read_bytes()
    assert json.loads(original_note)["source"] == str(source.resolve())
    assert install(fixture) == probe
    assert note.read_bytes() == original_note

    async def forbidden(*args):
        pytest.fail("an imported baseline must not generate replacement answers")

    assert not asyncio.run(
        QualityProbeCoordinator(monitor, forbidden).check(
            policy_step=0, policy_snapshot_id="initial"
        )
    )
    assert install(fixture, optimizer_step=2, policy_snapshot_id="updated") == probe
    assert saved.read_bytes() == source.read_bytes()
    assert note.read_bytes() == original_note


@pytest.mark.parametrize(
    "change",
    [
        {"policy_step": 1},
        {"policy_step": False},
        {"panel_id": "different"},
        {"condition_id": "different"},
        {"policy_snapshot_id": "different"},
        {"source_question_count": 0},
        {"source_question_count": 2},
        {"source_question_count": True},
        {"metrics": {}},
        {"metrics": {"success": None}},
        {"metrics": {"success": 0.4}},  # Passes retention but not the T0 floor.
    ],
)
def test_invalid_import_is_rejected_before_baseline_installation(fixture, change):
    source, monitor, _, probe = fixture
    source.write_text(json.dumps(asdict(replace(probe, **change))))
    with pytest.raises(ValueError):
        install(fixture)
    assert not (monitor.root / "probe-00000000.json").exists()
    assert not (monitor.root / "initial-probe-import.json").exists()


def test_panel_identity_and_policy_minimum_are_both_checked(fixture):
    source, monitor, panel, _ = fixture
    with pytest.raises(ValueError):
        import_initial_quality_probe(
            source,
            monitor=monitor,
            panel=replace(panel, panel_id="other"),
            optimizer_step=0,
            policy_snapshot_id="initial",
        )
    monitor.policy = replace(monitor.policy, minimum_source_questions=2)
    with pytest.raises(ValueError):
        install(fixture)


def test_population_alias_is_not_an_extra_independent_source(fixture):
    source, monitor, panel, _ = fixture
    panel = replace(
        panel,
        slots=(*panel.slots, PanelSlot("another-occurrence", "synthetic", "alias", "source")),
    )
    import_initial_quality_probe(
        source, monitor=monitor, panel=panel, optimizer_step=0, policy_snapshot_id="initial"
    )


def test_resume_cannot_install_previously_unbound_t0(fixture):
    source, monitor, _, _ = fixture
    with pytest.raises(ValueError):
        install(fixture, optimizer_step=2, policy_snapshot_id="updated")
    # A collected or copied file alone does not establish the initial import binding.
    (monitor.root / "probe-00000000.json").write_bytes(source.read_bytes())
    with pytest.raises(ValueError):
        install(fixture, optimizer_step=2, policy_snapshot_id="updated")


@pytest.mark.parametrize("step", [0, 2])
def test_different_baseline_never_overwrites_original(fixture, step):
    source, monitor, _, probe = fixture
    install(fixture)
    saved = (monitor.root / "probe-00000000.json").read_bytes()
    source.write_text(json.dumps(asdict(replace(probe, evidence_id="replacement"))))
    with pytest.raises(ValueError):
        install(fixture, optimizer_step=step)
    assert (monitor.root / "probe-00000000.json").read_bytes() == saved


def test_formal_import_requires_policy_and_panel_before_run_directory_creation(tmp_path):
    from types import SimpleNamespace

    from skillev_private.experiments.bayesian_improve_training import run_coordinator

    root = tmp_path / "new-run"
    with pytest.raises(ValueError):
        asyncio.run(
            run_coordinator(
                config=SimpleNamespace(steps=250),
                bindings=None,
                profile=None,
                root=root,
                resume=None,
                topology=None,
                initial_quality_probe=tmp_path / "original.json",
            )
        )
    assert not root.exists()
