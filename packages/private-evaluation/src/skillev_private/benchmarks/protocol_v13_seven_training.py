"""Seven-domain training condition; legacy eight-domain files are read-only sources.

B=28: one source occurrence per allowed domain, four independent rollouts.
There is no eighth occurrence or rotating overweight domain. Source lanes retain
their predeclared seed-zero order; exhausting a
lane cycles it with new occurrence identities, never new labels or shared state.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from skillev.evaluation.current_iid.protocol13.catalog import Protocol13Benchmark

from .protocol_v13_training import Protocol13TrainingRecord

SEVEN_DOMAIN_CONDITION = "seven-domain-balanced-7x4-seed0@2"
QUESTIONS_PER_STEP = 7
TRAJECTORIES_PER_QUESTION = 4
EFFECTIVE_BATCH_SIZE = QUESTIONS_PER_STEP * TRAJECTORIES_PER_QUESTION
SEVEN_TRAINING_DOMAINS = (
    Protocol13Benchmark.HOTPOT_QA,
    Protocol13Benchmark.TRIVIA_QA,
    Protocol13Benchmark.AIME_2026,
    Protocol13Benchmark.HEALTHBENCH,
    Protocol13Benchmark.ALF_WORLD,
    Protocol13Benchmark.MBPP_PLUS,
    Protocol13Benchmark.HUMAN_EVAL,
)
SIX_TRAINING_DOMAINS = SEVEN_TRAINING_DOMAINS[:-1]


def domain_schedule_condition(domains: tuple[Protocol13Benchmark, ...]) -> str:
    if domains == SEVEN_TRAINING_DOMAINS:
        return SEVEN_DOMAIN_CONDITION
    return "balanced-domain-subset-4x-seed0@1:" + ",".join(d.value for d in domains)


def load_seven_domain_training_sources(path: Path) -> tuple[Protocol13TrainingRecord, ...]:
    """Read current lanes without requiring the retired eighth-domain population.

    Legacy acquisition files can still supply the seven unchanged source lanes;
    their WebShop rows are excluded, never hydrated or scheduled. A seven-only
    source file does not need dummy WebShop records to satisfy a historical gate.
    """
    with path.open(encoding="utf-8") as stream:
        records = (
            Protocol13TrainingRecord.from_value(json.loads(line)) for line in stream if line.strip()
        )
        return tuple(r for r in records if r.episode.benchmark in SEVEN_TRAINING_DOMAINS)


def seven_domain_training_trajectories(
    records: tuple[Protocol13TrainingRecord, ...],
    *,
    steps: int,
    domain_starts: Mapping[int, tuple[Protocol13Benchmark, ...]] | None = None,
) -> tuple[Protocol13TrainingRecord, ...]:
    if type(steps) is not int or not 1 <= steps <= 250:
        raise ValueError("seven-domain training requires 1 through 250 complete steps")
    starts = dict(domain_starts or {1: SEVEN_TRAINING_DOMAINS})
    if 1 not in starts or any(type(step) is not int or not 1 <= step <= steps for step in starts):
        raise ValueError("domain schedule must start at the first optimizer step")
    for domains in starts.values():
        if not domains or tuple(d for d in SEVEN_TRAINING_DOMAINS if d in domains) != domains:
            raise ValueError("domains must be a nonempty canonical subset without duplicates")
    required = set().union(*starts.values())
    lanes = {
        domain: tuple(record for record in records if record.episode.benchmark is domain)
        for domain in SEVEN_TRAINING_DOMAINS
        if domain in required
    }
    if any(not lane for lane in lanes.values()):
        raise ValueError("seven-domain training requires a source lane for every allowed domain")
    cursors: Counter[Protocol13Benchmark] = Counter()
    repeats: Counter[tuple[Protocol13Benchmark, str]] = Counter()
    result = []
    for step in range(steps):
        active = starts[max(start for start in starts if start <= step + 1)]
        questions = []
        for slot, domain in enumerate(SEVEN_TRAINING_DOMAINS):
            if domain not in active:
                continue
            lane = lanes[domain]
            source = lane[cursors[domain] % len(lane)]
            cursors[domain] += 1
            key = domain, source.episode.source_id
            occurrence = repeats[key]
            repeats[key] += 1
            # Keep the original source-lane coordinates across a domain removal.
            # Membership is declared separately; retained tasks are not renamed
            # or shifted to an earlier source when the batch becomes smaller.
            episode_id = f"{SEVEN_DOMAIN_CONDITION}/step-{step + 1:03d}/question-{slot:02d}"
            episode = replace(
                source.episode,
                episode_id=episode_id,
                repeat_ordinal=occurrence,
                block_position=step,
                optimizer_step=step + 1,
                global_position=step * QUESTIONS_PER_STEP + slot,
            )
            questions.append(
                replace(source, episode=episode, input=replace(source.input, task_id=episode_id))
            )
        for rollout in range(TRAJECTORIES_PER_QUESTION):
            for record in questions:
                task_id = f"{record.episode.episode_id}/rollout-{rollout:02d}"
                result.append(
                    replace(
                        record,
                        episode=replace(record.episode, episode_id=task_id),
                        input=replace(record.input, task_id=task_id),
                    )
                )
    return tuple(result)
