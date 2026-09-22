"""v7 author evidence: head+tail excerpts, output-finish facts, contrasts, family choice."""

import asyncio
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from skillev.contracts.canonical import canonical_json
from skillev.contracts.evosteer import EvoTask
from skillev.evolution.evosteer_author import FrozenSkillAuthor, SkillAuthorConfig
from skillev.evolution.validated_admission import AdmissionConfig, AdmissionLedger, SkillEntry
from tests.evosteer.test_evosteer_author import Model, assessed, budget, trajectory

# ---------------------------------------------------------------------------
# Author prompt evidence (no torch required).
# ---------------------------------------------------------------------------


def executed(
    sample,
    *,
    task_id="q",
    reward,
    output,
    output_tokens,
    limit=64,
    metadata=None,
    extra_result=None,
):
    """One solver execution selected as output, with the runtime's saved event shape."""
    source = trajectory(sample, task_id, reward)
    result = {
        "output": output,
        "usage": {"output_tokens": output_tokens, "model_calls": 1},
        "metadata": metadata or {},
        **(extra_result or {}),
    }
    history = [
        {
            "sequence": 1,
            "action": {"kind": "ADD_AGENT", "node_id": "n0", "role_id": "solver"},
            "observation": {
                "node_id": "n0",
                "node_request": {
                    "runtime_id": "runtime-secret",
                    "role": {
                        "role_id": "solver",
                        "instruction": "Solve the task.",
                        "model_maximum": {"output_tokens": limit, "model_calls": 1},
                    },
                    "skills": [{"skill_id": "s", "body": "SKILL-BODY-SECRET"}],
                },
                "node_result": result,
                "output_version": 1,
                "output_changed": False,
                "reservation_id": "BUDGET-SECRET",
            },
        },
        {
            "sequence": 2,
            "action": {"kind": "SET_OUTPUT", "node_id": "n0"},
            "observation": {"selected_output_node": "n0"},
        },
        {
            "sequence": 3,
            "action": {"kind": "STOP"},
            "observation": {"final_output": output, "output_node_id": "n0"},
        },
    ]
    terminal = {
        "stopped": True,
        "poisoned": False,
        "graph": {
            "nodes": [
                {
                    "node_id": "n0",
                    "role_id": "solver",
                    "skill_ids": [],
                    "output": output,
                    "output_version": 1,
                }
            ],
            "edges": [],
            "output_node_id": "n0",
        },
        "history": history,
    }
    return assessed(
        replace(source, terminal_state_json=canonical_json(terminal), output=output, risk=None)
    )


def material(model):
    return json.loads(model.calls[0][0].split("\n", 1)[1])


def evidence(model):
    return {item["sample_id"]: item for item in material(model)["scored_public_trajectories"]}


def propose(items, **config):
    model = Model()
    author = FrozenSkillAuthor(model, budget=budget(), config=SkillAuthorConfig(**config))
    author.propose(tuple(items), family="math", window_id="window", seed=0)
    return model, author


def test_long_output_excerpt_keeps_its_ending_and_counts_the_omission():
    # A derivation cut off at the output limit is only visible at its END.
    ending = "so the remaining case analysis gives"
    output = "Setup: " + "x" * 5000 + ending
    model, _ = propose(
        (executed("cut", reward=0.0, output=output, output_tokens=64),),
        max_public_text_chars=500,
        input_limit=100_000,
    )
    shown = evidence(model)["cut"]["generated_output"]
    assert shown.startswith("Setup: " + "x" * 293)  # 60% head
    assert shown.endswith(ending)  # 40% tail
    assert f"[... {len(output) - 500} characters omitted ...]" in shown
    assert "[excerpt]" not in model.calls[0][0]
    # Short texts are never marked.
    assert evidence(model)["cut"]["public_task"] == trajectory().task.prompt


def test_prompt_shrinking_keeps_every_output_ending_and_the_unexcerpted_facts():
    items = [
        executed(
            f"run-{index}",
            task_id=f"t{index}",
            reward=float(index % 2),
            output="Begin. " + "step " * 1200 + f"E{index}",
            output_tokens=64,
        )
        for index in range(4)
    ]
    model, author = propose(items, input_limit=6000, max_trajectories=4)
    assert len(model.calls[0][0].encode("utf-8")) <= 6000
    limits = material(model)["excerpt_limits"]
    assert limits["text_characters"] < SkillAuthorConfig().max_public_text_chars
    shown = evidence(model)
    assert set(shown) == set(author.reports[0].selected_sample_ids)
    for index, item in enumerate(items):
        row = shown[item.sample_id]
        assert row["generated_output"].endswith(f"E{index}")
        # Facts come from the full output and history, not from the excerpt.
        assert row["output_facts"]["output_characters"] == len(item.output)
        assert row["output_facts"]["output_node_stopped_at_output_token_limit"] is True


def test_output_facts_separate_a_cut_off_output_from_a_finished_one():
    cut = executed(
        "cut", task_id="z", reward=0.0, output="Let n = 3. Check " + "case " * 120, output_tokens=64
    )
    done = executed(
        "done",
        task_id="z",
        reward=1.0,
        output="Derivation. " + "step " * 100 + "\nFinal: \\boxed{7}",
        output_tokens=20,
    )
    short = executed("short", task_id="s", reward=1.0, output="Paris", output_tokens=2)
    code = executed(
        "code", task_id="c", reward=0.0, output="```python\ndef f(x):\n    return", output_tokens=9
    )
    model, _ = propose((cut, done, code, short), max_trajectories=4)
    facts = {key: item["output_facts"] for key, item in evidence(model).items()}
    assert facts["cut"] == {
        "output_characters": len(cut.output),
        "final_answer_marker_in_tail": False,
        "unclosed_code_fence": False,
        "output_node_stopped_at_output_token_limit": True,
        "node_executions": 1,
        "node_executions_stopped_at_output_token_limit": 1,
    }
    assert facts["done"]["final_answer_marker_in_tail"] is True
    assert facts["done"]["output_node_stopped_at_output_token_limit"] is False
    assert facts["done"]["node_executions_stopped_at_output_token_limit"] == 0
    assert facts["code"]["unclosed_code_fence"] is True
    # A short output is shown whole; its missing marker says nothing.
    assert facts["short"]["final_answer_marker_in_tail"] is None
    assert facts["code"]["final_answer_marker_in_tail"] is None
    assert facts["code"]["output_node_stopped_at_output_token_limit"] is False
    # The author is told what the facts and the omission marker mean.
    instructions = model.calls[0][0].split("\n", 1)[0]
    assert "output_facts" in instructions
    assert "characters omitted" in instructions


@pytest.mark.parametrize(
    "metadata",
    [
        {"execution_outcome": {"status": "answered", "failure_kind": "truncated"}},
        {"finish_reason": "length"},
    ],
)
def test_executor_reported_length_stop_counts_even_below_the_role_limit(metadata):
    source = executed("reported", reward=0.0, output="partial", output_tokens=10, metadata=metadata)
    model, _ = propose((source,))
    facts = evidence(model)["reported"]["output_facts"]
    assert facts["output_node_stopped_at_output_token_limit"] is True
    assert facts["node_executions_stopped_at_output_token_limit"] == 1


def test_missing_usage_or_output_node_is_unknown_rather_than_claimed_finished():
    # The legacy fixture saves a node result without usage and no graph.
    model, _ = propose((trajectory(),))
    facts = evidence(model)["success"]["output_facts"]
    assert facts["output_node_stopped_at_output_token_limit"] is None
    assert facts["node_executions"] == 1
    assert facts["node_executions_stopped_at_output_token_limit"] == 0


def test_output_facts_ignore_evaluator_payloads_and_leak_no_request_text():
    first = executed(
        "first",
        task_id="a",
        reward=0.0,
        output="Answer: 12",
        output_tokens=5,
        extra_result={"gold": "GOLD-ANSWER-ONE", "hidden_tests": ["assert HIDDEN-TEST"]},
    )
    second = executed(
        "second",
        task_id="b",
        reward=0.0,
        output="Answer: 12",
        output_tokens=5,
        extra_result={"gold": "GOLD-ANSWER-TWO", "hidden_tests": ["assert HIDDEN-TEST-2"]},
    )
    model, _ = propose((first, second))
    shown = evidence(model)
    assert shown["first"]["output_facts"] == shown["second"]["output_facts"]
    text = model.calls[0][0]
    for secret in (
        "GOLD-ANSWER",
        "HIDDEN-TEST",
        "SKILL-BODY-SECRET",
        "BUDGET-SECRET",
        "runtime-secret",
        "model_maximum",
    ):
        assert secret not in text


def test_stop_event_does_not_repeat_the_final_output():
    output = "UNIQUE-FINAL-OUTPUT-TEXT"
    model, _ = propose((executed("once", reward=1.0, output=output, output_tokens=5),))
    text = model.calls[0][0]
    # generated_output and the executed node's own result; not the STOP echo.
    assert text.count(output) == 2
    stop = evidence(model)["once"]["executed_history"][-1]
    assert stop["observation"] == {
        "final_output": "[identical to generated_output]",
        "output_node_id": "n0",
    }


# ---------------------------------------------------------------------------
# Trajectory selection (paper Appendix C contrast rule).
# ---------------------------------------------------------------------------


def test_every_mixed_task_contributes_a_contrast_before_unpaired_successes():
    _, author = propose(
        (
            trajectory("easy-1", "easy", 1.0),
            trajectory("easy-2", "easy-2", 1.0),
            trajectory("a-ok", "a", 1.0),
            trajectory("a-bad", "a", 0.0),
            trajectory("b-ok", "b", 0.75),
            trajectory("b-bad", "b", 0.25),
        ),
        max_trajectories=4,
    )
    # Largest reward gap first; previously only ONE contrast was kept and the
    # rest of the window was filled with the highest-reward runs.
    assert author.reports[0].selected_sample_ids == ("a-ok", "a-bad", "b-ok", "b-bad")


def test_window_without_a_same_task_contrast_still_shows_failures():
    _, author = propose(
        (
            trajectory("s1", "t1", 1.0),
            trajectory("s2", "t2", 1.0),
            trajectory("s3", "t3", 0.9),
            trajectory("f1", "t4", 0.0),
            trajectory("f2", "t5", 0.2),
        ),
        max_trajectories=4,
    )
    assert author.reports[0].selected_sample_ids == ("s1", "f1", "s2", "f2")


def test_fill_prefers_tasks_not_already_shown():
    _, author = propose(
        (
            trajectory("z-ok", "z", 1.0),
            trajectory("z-bad", "z", 0.0),
            trajectory("z-ok-again", "z", 1.0),
            trajectory("y-ok", "y", 0.9),
        ),
        max_trajectories=3,
    )
    assert author.reports[0].selected_sample_ids == ("z-ok", "z-bad", "y-ok")


def test_all_success_window_keeps_best_first_order():
    _, author = propose(
        (trajectory("b", "t2", 0.9), trajectory("a", "t1", 1.0), trajectory("c", "t3", 0.6)),
        max_trajectories=2,
    )
    assert author.reports[0].selected_sample_ids == ("a", "b")


# ---------------------------------------------------------------------------
# Author family choice in EvoSteerApplication._prepare (torch integration).
# ---------------------------------------------------------------------------


class RecordingAuthor:
    """Records the family each author window is opened for; proposes nothing."""

    def __init__(self):
        self.families = []

    def propose(self, trajectories, *, family, window_id, seed):
        assert isinstance(trajectories, tuple)
        self.families.append(family)
        return None


def family_binding(task_id, family, reward):
    from tests.evosteer.test_application import binding

    base = binding(task_id)

    def session(request):
        opened = base.session_factory(request)
        opened.evaluator = lambda output: reward
        return opened

    return replace(
        base,
        task=EvoTask(task_id, family, "Produce a checked result."),
        session_factory=session,
    )


def skill(skill_id, family):
    body = f"Procedure for {family}."
    return SkillEntry(
        skill_id, family, "sha256:" + hashlib.sha256(body.encode()).hexdigest(), body=body
    )


def chosen_family(families, bindings, *, skills=(), max_candidates=1):
    from skillev.evosteer_application import EvoSteerApplication
    from tests.evosteer.test_application import application

    source = application(candidate=False)
    config = replace(
        source.config, task_families=families, max_candidates_per_family=max_candidates
    )
    ledger = AdmissionLedger(AdmissionConfig(max_candidates_per_family=max_candidates))
    for entry in skills:
        ledger.propose(entry, author_window_id="initial")
    author = RecordingAuthor()
    app = EvoSteerApplication(source.policy, config, admission=ledger, author=author)
    prepared = app.prepare_batch(asyncio.run(app.collect_batch(tuple(bindings), batch_number=1)))
    assert author.families == [prepared.author_result["family"]]
    return author.families[0]


def test_author_tie_goes_to_the_failing_family_not_the_first_configured():
    # Config order used to pick the ceiling family (every run a success) first.
    assert (
        chosen_family(
            ("ceiling", "failing"),
            (family_binding("up", "ceiling", 1.0), family_binding("down", "failing", 0.0)),
        )
        == "failing"
    )


def test_fewest_skills_still_takes_precedence_over_failures():
    assert (
        chosen_family(
            ("ceiling", "failing"),
            (family_binding("up", "ceiling", 1.0), family_binding("down", "failing", 0.0)),
            skills=(skill("candidate-failing", "failing"),),
            max_candidates=2,
        )
        == "ceiling"
    )


def test_family_absent_from_the_batch_loses_its_tie():
    assert (
        chosen_family(("absent", "present"), (family_binding("only", "present", 1.0),))
        == "present"
    )


def test_equal_mean_reward_tie_goes_to_the_family_with_more_failed_runs():
    # "partial" never falls below the success threshold; "split" fails half its runs.
    assert (
        chosen_family(
            ("partial", "split"),
            (
                family_binding("half", "partial", 0.5),
                family_binding("win", "split", 1.0),
                family_binding("lose", "split", 0.0),
            ),
        )
        == "split"
    )


def alternating_binding(task_id, family):
    """A task whose runs alternate between success and failure (a same-task contrast)."""
    from itertools import cycle

    from tests.evosteer.test_application import binding

    base = binding(task_id)
    rewards = cycle((1.0, 0.0))

    def session(request):
        opened = base.session_factory(request)
        reward = next(rewards)
        opened.evaluator = lambda output: reward
        return opened

    return replace(
        base,
        task=EvoTask(task_id, family, "Produce a checked result."),
        session_factory=session,
    )


def test_a_family_with_same_task_contrasts_beats_one_that_only_fails():
    # All-failure evidence has no success to contrast with; mixed tasks do.
    assert (
        chosen_family(
            ("floor", "mixed"),
            (family_binding("never", "floor", 0.0), alternating_binding("both", "mixed")),
        )
        == "mixed"
    )


class ReportingAuthor(RecordingAuthor):
    """Earlier windows opened for ``stalled`` produced no candidate."""

    def __init__(self, stalled):
        super().__init__()
        self.reports = tuple(
            SimpleNamespace(family=stalled, status=status) for status in ("null", "call-error")
        )


def test_a_family_whose_windows_produced_nothing_yields_its_tie():
    from skillev.evosteer_application import EvoSteerApplication
    from tests.evosteer.test_application import application

    def choose(author):
        source = application(candidate=False)
        config = replace(source.config, task_families=("first", "second"))
        app = EvoSteerApplication(source.policy, config, author=author)
        bindings = (family_binding("a", "first", 0.0), family_binding("b", "second", 0.0))
        app.prepare_batch(asyncio.run(app.collect_batch(bindings, batch_number=1)))
        return author.families[0]

    assert choose(RecordingAuthor()) == "first"
    # Without the rotation key the same family would hold every window.
    assert choose(ReportingAuthor("first")) == "second"
