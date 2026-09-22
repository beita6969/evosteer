from skillev_private.benchmarks.humaneval_official import (
    HUMANEVAL_RESOURCE_LIMITS,
    current_humaneval_executor_contract,
)


def test_humaneval_executor_contract_records_interpreter_and_all_limits() -> None:
    contract = current_humaneval_executor_contract()
    limits = HUMANEVAL_RESOURCE_LIMITS
    assert contract.profile_id == "humaneval-python-isolated@2"
    assert contract.benchmark == "humaneval"
    assert contract.python_version
    assert contract.wall_timeout_seconds == str(limits.wall_timeout_seconds)
    assert contract.cpu_seconds == limits.cpu_seconds
    assert contract.address_space_bytes == limits.address_space_bytes
    assert contract.process_count == limits.process_count
    assert contract.file_size_bytes == limits.file_size_bytes
    assert contract.open_file_count == limits.open_file_count
    assert contract.test_profile == "openai-humaneval-original-tests@1"
