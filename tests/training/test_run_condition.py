import pytest

from skillev.training.run_condition import EffectiveRunCondition


def test_condition_copies_inputs_and_distinguishes_science_from_execution():
    science = {"sampling": {"temperature": 1.0}, "thinking": {"aime-2026": True}}
    first = EffectiveRunCondition.create(
        condition_id="first", scientific=science, execution={"gpu": "device-a"}
    )
    second = EffectiveRunCondition.create(
        condition_id="second", scientific=science, execution={"gpu": "device-b"}
    )
    first.require_same_science(second)
    assert first.differences(second)["scientific"] == []
    assert first.differences(second)["execution"]
    science["sampling"]["temperature"] = 0.5
    changed = EffectiveRunCondition.create(condition_id="first", scientific=science, execution={})
    with pytest.raises(ValueError):
        first.require_same_science(changed)
    assert first.scientific["sampling"]["temperature"] == 1.0
    assert first.differences(changed)["scientific"][0]["field"] == "scientific/sampling/temperature"
    assert EffectiveRunCondition.from_value(first.to_value()) == first
