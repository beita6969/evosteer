import json
from pathlib import Path

from skillev_private.direct_reference.populations import load_skillflow_iid_cases

from scripts import generate_qwen35_direct_swe_repo_agent as repo_agent
from skillev.evaluation.direct_baseline import DirectBenchmark
from skillev.evaluation.direct_baseline.protocol import load_direct_reference_protocol


def _task(index: int = 0) -> repo_agent.SWERepositoryTask:
    return repo_agent.SWERepositoryTask(
        task_id=f"task-{index}",
        instance_id=f"project__issue-{index}",
        repo="owner/project",
        base_commit=f"revision-{index}",
        problem_statement="Correct the public behavior.",
    )


def _outcome(task: repo_agent.SWERepositoryTask, state: str, attempt: int = 1):
    patch = "diff --git a/a.py b/a.py" if state == "candidate" else None
    error = "EpisodeInfrastructureError" if state == "infrastructure_failure" else None
    return repo_agent.EpisodeOutcome(
        task_id=task.task_id,
        instance_id=task.instance_id,
        attempt=attempt,
        state=state,
        model_patch=patch,
        infrastructure_error=error,
        endpoint_index=0,
        elapsed_seconds=1.0,
        steps=1,
        supervisor_calls=1,
        transcript=(),
    )


def test_repo_agent_model_question_is_answer_blind() -> None:
    visible = repo_agent._model_question(_task())

    assert set(visible) == {"question", "task_type", "context", "extra"}
    assert "answer" not in visible
    assert "code_files" not in visible
    assert visible["extra"] == {
        "instance_id": "project__issue-0",
        "repo": "owner/project",
        "base_commit": "revision-0",
    }


def test_formal_swe_protocol_selects_repository_agent_contract() -> None:
    protocol = load_direct_reference_protocol(
        Path("configs/evaluation/qwen35_skillflow_direct_reference.yaml")
    )
    spec = protocol.benchmark(DirectBenchmark.SWE_BENCH)

    assert spec.prompt_profile == "qwen-direct-readonly-repository-agent@1"
    assert spec.decoding_profile == "qwen35-repository-agent-supervisor@1"
    assert spec.parser_profile == "unified-diff@2"


def test_repo_agent_retries_only_infrastructure_failures() -> None:
    tasks = tuple(_task(index) for index in range(4))
    latest = {
        tasks[0].task_id: _outcome(tasks[0], "candidate"),
        tasks[1].task_id: _outcome(tasks[1], "candidate_failure"),
        tasks[2].task_id: _outcome(tasks[2], "infrastructure_failure"),
        tasks[3].task_id: _outcome(tasks[3], "infrastructure_failure", attempt=3),
    }

    assert repo_agent._pending_tasks(tasks, latest, max_attempts=3) == (tasks[2],)


def test_repo_agent_rejects_repository_path_traversal(tmp_path: Path) -> None:
    assert repo_agent._repository_path(tmp_path, "owner/project") == tmp_path / "owner__project"

    for value in ("../project", "owner/../project", "/project", "owner/project/extra"):
        try:
            repo_agent._repository_path(tmp_path, value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe repository identity accepted: {value}")


def test_repo_agent_binds_formal_seed_to_upstream_requests() -> None:
    calls: list[dict[str, object]] = []

    class FakeCompletions:
        def create(self, *args: object, **kwargs: object) -> object:
            calls.append(dict(kwargs))
            return object()

    seeded = repo_agent._SeededCompletions(FakeCompletions(), 42)

    seeded.create(model="qwen35-direct-base")
    seeded.create(model="qwen35-direct-base", seed=42)

    assert [call["seed"] for call in calls] == [42, 42]


def test_repo_agent_rejects_conflicting_upstream_seed() -> None:
    class FakeCompletions:
        def create(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("conflicting request must not reach the client")

    seeded = repo_agent._SeededCompletions(FakeCompletions(), 42)

    try:
        seeded.create(model="qwen35-direct-base", seed=7)
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting upstream seed was accepted")


def test_direct_repository_tool_path_is_workspace_bounded(tmp_path: Path) -> None:
    source = tmp_path / "src" / "module.py"
    source.parent.mkdir()
    source.write_text("value = 1\n", encoding="utf-8")

    assert (
        repo_agent._validated_direct_tool_path(tmp_path, "src/module.py", mutating=True)
        == source.resolve()
    )

    for path in ("../outside.py", "tests/test_module.py", ".git/config"):
        try:
            repo_agent._validated_direct_tool_path(tmp_path, path, mutating=True)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe direct edit path accepted: {path}")

    outside = tmp_path.parent / "outside.py"
    outside.write_text("private = True\n", encoding="utf-8")
    try:
        repo_agent._validated_direct_tool_path(tmp_path, outside, mutating=False)
    except ValueError:
        pass
    else:
        raise AssertionError("direct view accepted an existing path outside the workspace")


def test_direct_view_handler_applies_workspace_boundary(tmp_path: Path) -> None:
    source = tmp_path / "src" / "module.py"
    source.parent.mkdir()
    source.write_text("value = 1\n", encoding="utf-8")
    outside = tmp_path.parent / "private.py"
    outside.write_text("target = True\n", encoding="utf-8")

    class BaseEnvironment:
        def _handle_view_file(self, args: dict[str, object]) -> str:
            return str(args["path"])

    environment_type = repo_agent._isolated_environment_type(BaseEnvironment)
    environment = object.__new__(environment_type)
    environment._repo_path = str(tmp_path)

    assert environment._handle_view_file({"path": "src/module.py"}) == str(source.resolve())
    assert "[ERROR]" in environment._handle_view_file({"path": str(outside)})


def test_direct_raw_tools_disable_swe_orchestration_hints() -> None:
    class BaseEnvironment:
        def _swe_memory_summary(self, *args: object, **kwargs: object) -> str:
            return "orchestrated memory"

        def _swe_issue_member_mentions(self, *args: object, **kwargs: object) -> list[object]:
            return [("Class", "member")]

    environment_type = repo_agent._isolated_environment_type(BaseEnvironment)
    environment = object.__new__(environment_type)
    environment._skillev_direct_raw_tools = True

    assert environment._swe_memory_summary() == ""
    assert environment._swe_issue_member_mentions() == []

    environment._skillev_direct_raw_tools = False
    assert environment._swe_memory_summary() == "orchestrated memory"
    assert environment._swe_issue_member_mentions() == [("Class", "member")]


def test_direct_readonly_candidate_requires_one_unified_diff() -> None:
    patch = "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new"

    assert repo_agent._readonly_patch_candidate(f"```diff\n{patch}\n```") == patch
    assert repo_agent._readonly_patch_candidate("I need another source view.") is None
    assert (
        repo_agent._readonly_patch_candidate(
            f"```diff\n{patch}\n```\n```diff\n{patch.replace('+new', '+other')}\n```"
        )
        is None
    )


def test_swe_population_retains_public_repo_agent_fields(tmp_path: Path) -> None:
    population = tmp_path / "iid.json"
    rows = [
        {
            "question": f"Public issue {index}",
            "answer": "private fixture patch",
            "task_type": "code_generation",
            "context": [],
            "code_files": {"module.py": "value = 1\n"},
            "extra": {
                "source": "SWE-bench",
                "instance_id": f"project__issue-{index}",
                "repo": "owner/project",
                "base_commit": f"revision-{index}",
                "split": "test",
            },
        }
        for index in range(128)
    ]
    population.write_text(json.dumps(rows), encoding="utf-8")
    protocol = load_direct_reference_protocol(
        Path("configs/evaluation/qwen35_skillflow_direct_reference.yaml")
    )

    cases = load_skillflow_iid_cases(
        population,
        protocol=protocol,
        include=frozenset({DirectBenchmark.SWE_BENCH}),
    )

    metadata = cases[0].private_metadata
    assert metadata["instance_id"] == "project__issue-0"
    assert metadata["repo"] == "owner/project"
    assert metadata["base_commit"] == "revision-0"
    assert metadata["problem_statement"] == "Public issue 0"
    assert "pinned repository snapshot" in cases[0].public_task.messages[-1]["content"]
