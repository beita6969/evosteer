"""A resumed fresh run cannot exchange its original accepted A0 condition."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from skillev_private.experiments.fresh_restart import (
    load_fresh_config,
    require_run_iid_baselines,
)

from skillev.rollout.readonly_collection import evaluation_isolation


def accepted_baselines(tmp_path, config):
    locations = {}
    for arm in ("skills-off", "initial-library"):
        directory = tmp_path / arm
        directory.mkdir()
        locations[arm] = str(directory)
        expanded = {
            "architecture_id": "synthetic-architecture",
            "panel": {"source": "synthetic-readonly"},
            "controls": {
                "formal": config.expanded_value(),
                "sampling": config.sampling_config.to_value(),
            },
            "policy_snapshot": {"snapshot_id": "base0"},
            "library_snapshot": {"arm": arm},
            "acceptance_rules": {},
        }
        (directory / "expanded-controls-private.json").write_text(json.dumps(expanded))
        (directory / "summary.json").write_text(
            json.dumps(
                {
                    "record_kind": "iid-evaluation",
                    "policy_step": 0,
                    "policy_snapshot_id": "base0",
                    "arm": arm,
                    "baseline_accepted": True,
                    "execution_validation": "live-components-unchanged",
                    **evaluation_isolation(),
                    "domains": {
                        d: {"planned": 1, "completed": 1, "accepted": True} for d in config.domains
                    },
                }
            )
        )
    path = tmp_path / "iid-baselines-private.json"
    path.write_text(json.dumps(locations))
    return path


def test_fresh_and_resume_share_original_a0_controls(tmp_path):
    config = load_fresh_config(Path("configs/training/bayesianimprove_fresh_restart.yaml"))
    path = accepted_baselines(tmp_path, config)
    first = require_run_iid_baselines(requested=path, root=tmp_path, config=config, resuming=False)
    # Actual initialization verification is tested with a real dual-LoRA CPU
    # application in test_prepare_fresh_restart; here only its read-only binding.
    start = {"format": "fresh-application-start@1", "failures": [], "unverified": []}
    (tmp_path / "fresh-application-start.json").write_text(json.dumps(start))
    resumed = require_run_iid_baselines(requested=None, root=tmp_path, config=config, resuming=True)
    assert first == resumed
    changed = replace(config, max_action_tokens=1024)
    with pytest.raises(ValueError):
        require_run_iid_baselines(requested=None, root=tmp_path, config=changed, resuming=True)
    replacement = tmp_path / "different.json"
    replacement.write_text(json.dumps({"skills-off": "elsewhere", "initial-library": "elsewhere"}))
    with pytest.raises(ValueError):
        require_run_iid_baselines(
            requested=replacement, root=tmp_path, config=config, resuming=True
        )


def test_old_run_cannot_become_fresh_by_supplying_a_new_a0(tmp_path):
    config = load_fresh_config(Path("configs/training/bayesianimprove_fresh_restart.yaml"))
    path = accepted_baselines(tmp_path, config)
    with pytest.raises(FileNotFoundError):
        require_run_iid_baselines(requested=path, root=tmp_path, config=config, resuming=True)
    (tmp_path / "fresh-application-start.json").write_text(
        json.dumps(
            {
                "format": "fresh-application-start@1",
                "failures": ["optimizer_step_zero"],
                "unverified": [],
            }
        )
    )
    with pytest.raises(ValueError):
        require_run_iid_baselines(requested=None, root=tmp_path, config=config, resuming=True)
