"""Explicit small panels use public source order, never outcome-ranked selection."""

import pytest
from skillev_private.evaluation.integrity_sources import SourcePanel, select_source_panel

from skillev.evaluation.input_metric_contracts import PublicTaskView
from skillev.evaluation.integrity_pipeline import FrozenPanel


def test_declared_prefix_panel_does_not_require_a_canary_population():
    entries = tuple(
        PublicTaskView.from_record(f"case-{i}", "aime-2026", {"problem": "Synthetic task"})
        for i in range(4)
    )
    source = SourcePanel(
        FrozenPanel(entries, "development-exposed", "synthetic"),
        {},
        {},
        {},
        {e.task_id: [{"public_paragraph": "unchanged"}] for e in entries},
    )
    selected = select_source_panel(source, {"aime-2026": 3})
    assert selected.panel.entries == entries[:3]
    assert selected.panel.sample_counts == (("aime-2026", 3),)
    assert selected.public_input_receipts == {
        e.task_id: source.public_input_receipts[e.task_id] for e in entries[:3]
    }
    selected.panel.validate(canary=False)
    with pytest.raises(ValueError):
        select_source_panel(source, {"aime-2026": 5})
    with pytest.raises(ValueError):
        select_source_panel(source, {"unknown": 2})


def test_retired_webshop_record_is_readable_but_cannot_enter_a_new_iid_panel():
    entry = PublicTaskView.from_record("historical", "webshop", {"task": "Synthetic task"})
    panel = FrozenPanel((entry,), "historical", "synthetic", (("webshop", 1),))
    with pytest.raises(ValueError):
        panel.validate(canary=True)
    source = SourcePanel(panel, {}, {}, {})
    with pytest.raises(ValueError):
        select_source_panel(source, {"webshop": 1})
