from __future__ import annotations

import pytest

from skillev.rollout import CanonicalInitialContextAssembler
from skillev.rollout.prompt_profiles import InitialContextProfile, validate_initial_context_profile
from tests.rollout.engine_fakes import ByteTokenizer, default_request

NO_SKILL_PROFILES = (
    InitialContextProfile.SKILLEV_COLD_START,
    InitialContextProfile.BENCHMARK_NATIVE_NO_SKILL,
)


@pytest.mark.parametrize("profile", NO_SKILL_PROFILES)
@pytest.mark.parametrize(("active", "retrieved"), [(("skill",), ()), ((), ("skill",))])
def test_no_skill_profile_rejects_retrieved_and_active_skills(profile, active, retrieved) -> None:
    validate_initial_context_profile(
        profile=profile,
        active_skill_ids=(),
        retrieved_skill_ids=(),
    )
    with pytest.raises(ValueError):
        validate_initial_context_profile(
            profile=profile,
            active_skill_ids=active,
            retrieved_skill_ids=retrieved,
        )


@pytest.mark.parametrize("profile", NO_SKILL_PROFILES)
def test_direct_assembler_also_enforces_no_skill_profile(profile) -> None:
    request = default_request()
    assembler = CanonicalInitialContextAssembler(maximum_h0_tokens=20_000)
    with pytest.raises(ValueError):
        assembler.assemble(
            task=request.task,
            retrieved_skills=request.retrieved_skills,
            active_skill_ids=request.active_skill_ids,
            library_version=request.library_version,
            tokenizer=ByteTokenizer(),
            profile=profile,
        )


@pytest.mark.parametrize("profile", NO_SKILL_PROFILES)
def test_no_skill_context_keeps_the_task_and_interfaces_without_skill_dependency(profile) -> None:
    request = default_request()
    assembled = CanonicalInitialContextAssembler(maximum_h0_tokens=20_000).assemble(
        task=request.task,
        retrieved_skills=(),
        active_skill_ids=(),
        library_version=request.library_version,
        tokenizer=ByteTokenizer(),
        profile=profile,
    )
    assert request.task.query in assembled.text
    assert "Available Actions" in assembled.text
    assert '"answer"' in assembled.text
    assert "Retrieved Skills" not in assembled.text
    assert "retrieved skill" not in assembled.text.casefold()
    assert "strategy comes from" not in assembled.text
    assert not assembled.contract.retrieved_skill_ids
    assert not assembled.contract.active_skill_ids
