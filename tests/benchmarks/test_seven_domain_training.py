from __future__ import annotations

import asyncio
import json
from collections import Counter

import pytest
from skillev_private.benchmarks.protocol_v13_seven_training import (
    SEVEN_DOMAIN_CONDITION,
    SEVEN_TRAINING_DOMAINS,
    load_seven_domain_training_sources,
    seven_domain_training_trajectories,
)
from skillev_private.benchmarks.protocol_v13_training import build_protocol13_training_records
from skillev_private.benchmarks.session_hydration import hydrate_ordered

from skillev.training.rollout_workflow import AsyncResourceLimiter
from tests.benchmarks.test_protocol_v13_training import _sources


def test_current_source_loader_does_not_require_retired_domain(tmp_path):
    legacy = build_protocol13_training_records(_sources(9))
    seven = tuple(r for r in legacy if r.episode.benchmark in SEVEN_TRAINING_DOMAINS)
    path = tmp_path / "sources.jsonl"
    for population in (legacy, seven):
        path.write_text("".join(json.dumps(r.to_value()) + "\n" for r in population))
        loaded = load_seven_domain_training_sources(path)
        assert loaded == seven
        assert len(seven_domain_training_trajectories(loaded, steps=4)) == 112


def test_seven_domains_balanced_without_extra_or_label_source_aliasing():
    sources = build_protocol13_training_records(_sources(9))
    rows = seven_domain_training_trajectories(sources, steps=7)
    assert rows == seven_domain_training_trajectories(sources, steps=7)
    assert len(rows) == len({r.input.task_id for r in rows}) == 7 * 28
    for step in range(7):
        batch = rows[step * 28 : (step + 1) * 28]
        counts = Counter(r.episode.benchmark for r in batch)
        assert set(counts) == set(SEVEN_TRAINING_DOMAINS)
        assert set(counts.values()) == {4}
        assert {r.episode.optimizer_step for r in batch} == {step + 1}
        for row in batch:
            assert row.input.task_id.startswith(SEVEN_DOMAIN_CONDITION)
            original = next(r for r in sources if r.episode.source_id == row.episode.source_id)
            assert row.output == original.output
            assert row.input.query == original.input.query
    full = seven_domain_training_trajectories(sources, steps=250)
    assert len(full) == len({r.input.task_id for r in full}) == 7000
    assert Counter(r.episode.benchmark for r in full) == dict.fromkeys(SEVEN_TRAINING_DOMAINS, 1000)
    occurrences = Counter(r.episode.global_position for r in full)
    assert len(occurrences) == 1750
    assert set(occurrences.values()) == {4}
    assert set(occurrences) == set(range(1750))
    for step in range(250):
        batch = full[step * 28 : (step + 1) * 28]
        for slot, domain in enumerate(SEVEN_TRAINING_DOMAINS):
            replicas = batch[slot::7]
            assert all(row.episode.benchmark is domain for row in replicas)
            assert len({row.episode.source_id for row in replicas}) == 1
            assert len({row.episode.repeat_ordinal for row in replicas}) == 1
    with pytest.raises(ValueError):
        seven_domain_training_trajectories(
            tuple(r for r in sources if r.episode.benchmark is SEVEN_TRAINING_DOMAINS[0]), steps=4
        )


@pytest.mark.parametrize("cancel", [False, True])
def test_parallel_hydration_is_bounded_ordered_and_drains_cleanup(cancel):
    async def run():
        active = peak = 0
        cleaned = []
        entered, release = asyncio.Event(), asyncio.Event()

        async def create(position):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            entered.set()
            try:
                await release.wait()
                await asyncio.sleep((3 - position) / 100)
                return position
            finally:
                active -= 1
                cleaned.append(position)

        pending = asyncio.create_task(
            hydrate_ordered(range(4), create, limiter=AsyncResourceLimiter(2))
        )
        await entered.wait()
        if cancel:
            pending.cancel()
            await asyncio.sleep(0)
            assert not pending.done()
        release.set()
        if cancel:
            with pytest.raises(asyncio.CancelledError):
                await pending
        else:
            assert await pending == (0, 1, 2, 3)
        assert sorted(cleaned) == [0, 1, 2, 3]
        assert active == 0
        assert peak == 2

    asyncio.run(run())
