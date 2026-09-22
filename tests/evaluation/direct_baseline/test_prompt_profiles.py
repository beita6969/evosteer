from pathlib import Path

from skillev.evaluation.direct_baseline.config import DirectBenchmark
from skillev.evaluation.direct_baseline.prompts import (
    react_messages,
    render_interactive_messages,
    static_messages,
)
from skillev.evaluation.direct_baseline.static_tasks import load_decoding_profiles


def test_math_profile_has_large_reasoning_budget() -> None:
    profiles = load_decoding_profiles(
        Path("configs/evaluation/qwen35_skillflow_direct_reference.yaml")
    )
    assert profiles["qwen35-thinking-math@1"].enable_thinking
    assert profiles["qwen35-thinking-math@1"].max_new_tokens >= 8192


def test_static_prompts_do_not_request_json_actions() -> None:
    for benchmark in (
        DirectBenchmark.HOTPOT_QA,
        DirectBenchmark.AIME_2026,
        DirectBenchmark.MED_QA,
        DirectBenchmark.HUMAN_EVAL,
    ):
        rendered = "\n".join(message["content"] for message in static_messages(benchmark, "Q"))
        assert "JSON" not in rendered


def test_interactive_prompt_uses_native_action() -> None:
    rendered = react_messages(
        DirectBenchmark.WEB_SHOP,
        task="buy an item",
        observation="search is available",
    )
    assert "search[keywords]" in rendered[0]["content"]
    assert "Do not output JSON" in rendered[0]["content"]


def test_interactive_transcript_orders_each_observation_before_its_action() -> None:
    rendered = react_messages(
        DirectBenchmark.WEB_SHOP,
        task="buy an item",
        observation="results page",
        history=(("search[item]", "search page"),),
    )
    assert rendered[1]["content"].endswith(
        "State before action: search page\n\n"
        "Action: search[item]\n\n"
        "Environment returned state: search page\n\n"
        "Observation: results page"
    )


def test_native_benchmarks_have_distinct_public_contracts_and_history_window() -> None:
    systems = {
        benchmark: react_messages(benchmark, task="task", observation="observation")[0]["content"]
        for benchmark in (
            DirectBenchmark.WEB_SHOP,
            DirectBenchmark.ALF_WORLD,
            DirectBenchmark.SCIENCE_WORLD,
        )
    }
    assert len(set(systems.values())) == 3
    rendered = render_interactive_messages(
        "webshop-native-react@2",
        task="task",
        current_observation="current",
        history=(("search[a]", "one"), ("click[b]", "two")),
        history_window_steps=1,
    )
    assert "one" not in rendered[1]["content"]
    assert "two" in rendered[1]["content"]


def test_humaneval_profile_is_greedy() -> None:
    profiles = load_decoding_profiles(
        Path("configs/evaluation/qwen35_skillflow_direct_reference.yaml")
    )
    profile = profiles["qwen35-humaneval-deterministic@1"]
    assert profile.sampling_mode == "greedy"
    assert profile.temperature == 0
