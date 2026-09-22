"""Domain scheduling never changes the paired population or consults labels."""

from dataclasses import replace

import pytest
from skillev_private.evaluation.integrity_sources import SourcePanel, order_source_panel

from skillev.evaluation.input_metric_contracts import IID_BENCHMARKS, PublicTaskView
from skillev.evaluation.integrity_pipeline import FrozenPanel


def source_panel() -> SourcePanel:
    entries = (
        PublicTaskView.from_record("aime", "aime-2026", {"problem": "Synthetic calculation"}),
        PublicTaskView.from_record("home-one", "alfworld", {"task": "Synthetic household task"}),
        PublicTaskView.from_record(
            "health", "healthbench", {"prompt": [{"role": "user", "content": "Synthetic question"}]}
        ),
        PublicTaskView.from_record("home-two", "alfworld", {"task": "Another synthetic task"}),
    )
    return SourcePanel(
        FrozenPanel(entries, "synthetic", "no benchmark examples"),
        {entry.task_id: {"synthetic_label": index} for index, entry in enumerate(entries)},
        {},
        {"population": "synthetic"},
    )


def test_domain_order_preserves_every_public_entry_and_within_domain_order() -> None:
    source = source_panel()
    order = ("alfworld", *(name for name in IID_BENCHMARKS if name != "alfworld"))
    ordered = order_source_panel(source, order)
    assert ordered.panel.entries[:2] == (source.panel.entries[1], source.panel.entries[3])
    assert sorted(ordered.panel.entries, key=lambda entry: entry.task_id) == sorted(
        source.panel.entries, key=lambda entry: entry.task_id
    )
    assert ordered.targets is source.targets
    assert ordered.interactive is source.interactive
    assert ordered.provenance is source.provenance
    changed_labels = replace(source, targets={})
    assert order_source_panel(changed_labels, order).panel == ordered.panel


@pytest.mark.parametrize(
    "order",
    [IID_BENCHMARKS[:-1], (*IID_BENCHMARKS, "webshop"), ("unsupported", *IID_BENCHMARKS[1:])],
)
def test_domain_order_cannot_omit_duplicate_or_introduce_a_benchmark(
    order: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError):
        order_source_panel(source_panel(), order)
