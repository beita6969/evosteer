from skillev.evaluation.current_iid.rendering import render_current_iid_markdown


def test_renderer_does_not_call_local_qwen_health_official() -> None:
    rendered = render_current_iid_markdown(
        {
            "format": "skillev-current-iid-result@1",
            "benchmarks": {
                "healthbench": {
                    "condition_id": "healthbench-qwen@1",
                    "status": "target-undefined",
                    "coverage": {"planned": 128, "definitive": 128},
                    "metrics": {"local_qwen_judge_score": {"observed": 52.4}},
                }
            },
        }
    )
    assert "not an official GPT-4.1 score" in rendered
    assert "128/128; candidate=0; infra(g/s/e)=0/0/0" in rendered
