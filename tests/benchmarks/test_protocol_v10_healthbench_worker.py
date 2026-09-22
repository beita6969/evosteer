from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from skillev_private.benchmarks.protocol_v10_healthbench_worker import (
    _BoundedOpenAISampler,
    _rubrics,
)
from skillev_private.benchmarks.protocol_v10_workers import (
    OfficialHealthBenchProcessGrader,
    PrivateJSONWorker,
)


def _official_fixture(root: Path) -> None:
    root.mkdir()
    (root / "types.py").write_text(
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class SamplerResponse:\n"
        " response_text: str\n"
        " response_metadata: dict\n"
        " actual_queried_message_list: list\n",
        encoding="utf-8",
    )
    (root.parent / "openai.py").write_text(
        "from types import SimpleNamespace as NS\n"
        "class OpenAI:\n"
        " def __init__(self, **kwargs):\n"
        "  self.chat=NS(completions=self)\n"
        " def create(self, **kwargs):\n"
        "  assert kwargs['extra_body'] == "
        "{'chat_template_kwargs': {'enable_thinking': False}}\n"
        '  message=NS(content=\'{"criteria_met":true,"explanation":"ok"}\')\n'
        "  return NS(choices=[NS(message=message)], usage=None)\n",
        encoding="utf-8",
    )
    (root / "healthbench_eval.py").write_text(
        "import json\n"
        "class RubricItem:\n"
        " def __init__(self, criterion, points, tags):\n"
        "  self.criterion, self.points, self.tags = criterion, points, tags\n"
        " @classmethod\n"
        " def from_dict(cls, value):\n"
        "  return cls(value['criterion'], value['points'], value['tags'])\n"
        "class HealthBenchEval:\n"
        " def grade_sample(self, prompt, response_text, example_tags, rubric_items):\n"
        "  grades=[]\n"
        "  for item in rubric_items:\n"
        "   result=self.grader_model([{'role':'user','content':item.criterion}])\n"
        "   parsed=json.loads(result.response_text)\n"
        "   grades.append({'points':item.points,'criteria_met':parsed['criteria_met']})\n"
        "  possible=sum(item.points for item in rubric_items if item.points > 0)\n"
        "  achieved=sum(item.points for item, grade in zip(rubric_items, grades) "
        "if grade['criteria_met'])\n"
        "  return {'overall_score':achieved/possible}, '', grades\n",
        encoding="utf-8",
    )


def test_healthbench_worker_keeps_private_rubric_in_grader_process(
    tmp_path: Path,
) -> None:
    official = tmp_path / "simple-evals"
    _official_fixture(official)
    worker_path = (
        Path(__file__).parents[2]
        / "packages/private-evaluation/src/skillev_private/benchmarks"
        / "protocol_v10_healthbench_worker.py"
    ).resolve()
    grader = OfficialHealthBenchProcessGrader(
        worker=PrivateJSONWorker(
            command=(
                str(Path(sys.executable)),
                str(worker_path),
                "--official-source-root",
                str(official.resolve()),
                "--grader-model",
                "fixture-grader",
                "--api-base-url",
                "http://127.0.0.1:30000/v1",
                "--request-timeout-seconds",
                "5",
            ),
            working_directory=tmp_path.resolve(),
            timeout_seconds=20,
        ),
        verifier_version="healthbench-fixture",
        private_cases={
            "health/task-1": {
                "grader_kind": "official-healthbench",
                "prompt": [{"content": "Question", "role": "user"}],
                "rubrics": [{"criterion": "Private criterion", "points": 1.0, "tags": []}],
            }
        },
    )

    result = asyncio.run(grader.grade("health/task-1", "Candidate response"))

    assert result.official_rubric_score == 1.0
    assert result.triggered_negative_rubric_count == 0


def test_healthbench_worker_accepts_qwen_local_rubric_route() -> None:
    class _Rubric:
        @classmethod
        def from_dict(cls, value: object) -> object:
            return value

    rubrics = [{"criterion": "Private criterion", "points": 1.0, "tags": []}]

    result = _rubrics(
        {
            "grader_kind": "healthbench-qwen35-local-simple-evals",
            "rubrics": rubrics,
        },
        _Rubric,
    )

    assert result == rubrics


@pytest.mark.parametrize("repair_budget", [1024, 4096])
def test_judge_repair_budget_is_explicit_and_does_not_change_first_attempt(repair_budget):
    requests = []

    def create(**request):
        requests.append(request)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"criteria_met":true}'))],
            usage=None,
        )

    sampler = _BoundedOpenAISampler(
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
        "frozen-base",
        SimpleNamespace,
        max_calls=6,
        repair_max_output_tokens=repair_budget,
    )
    a = [{"role": "user", "content": "criterion A"}]
    b = [{"role": "user", "content": "criterion B"}]
    sampler(a)
    sampler(b)
    sampler(a)
    sampler(b)
    assert [r["max_tokens"] for r in requests] == [1024, 1024, repair_budget, repair_budget]
    assert requests[0]["messages"] == a
    assert requests[2]["messages"][:-1] == a
    assert all(r["temperature"] == 0 and r["seed"] == 0 for r in requests)
    assert all(r["model"] == "frozen-base" for r in requests)
    with pytest.raises(RuntimeError):
        sampler(a)
    assert len(requests) == 4
