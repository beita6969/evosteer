"""One controlled CPU lifecycle; report reads metadata, not policy or optimizer."""

import asyncio
import json

import pytest

from skillev.training.skill_evidence_report import main, report_from_checkpoint
from tests.application.test_full_vertical_loop import build_application_fixture


def test_complete_application_checkpoint_exports_persisted_skill_report(
    tmp_path, training_backbone_config, monkeypatch
):
    fixture = build_application_fixture(tmp_path, training_backbone_config)
    asyncio.run(fixture.application.evolution_loop.run(fixture.run_plan))
    paths = list(tmp_path.glob("**/runtime_state.json"))
    path = max(paths, key=lambda item: json.loads(item.read_text())["optimizer_step"])
    before = path.read_bytes()
    report = report_from_checkpoint(path)
    assert report["optimizer_step"] == 3
    assert len(report["skill_coverage"]) == 3
    assert report["posterior_evidence_composition"]
    assert len(report["skill_lifecycle"]["skills"]) > len(fixture.initial_skill_ids)
    assert report["prequential_calibration"][-1]["optimizer_step"] == 3
    output = tmp_path / "skill-evidence-step3.json"
    monkeypatch.setattr(
        "sys.argv", ["report", "--runtime-state", str(path), "--output", str(output)]
    )
    main()
    assert json.loads(output.read_text()) == report
    assert path.read_bytes() == before
    with pytest.raises(FileExistsError):
        main()
    isolated = tmp_path / "incomplete"
    isolated.mkdir()
    (isolated / "runtime_state.json").write_bytes(before)
    with pytest.raises(ValueError):
        report_from_checkpoint(isolated / "runtime_state.json")
