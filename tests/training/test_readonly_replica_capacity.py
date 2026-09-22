"""Actual scheduler capacity can be lower than the advertised context length."""

import pytest
from skillev_private.experiments.readonly_replicas import measured_capacity, require_capacity


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {},
        {"context_length": 10000},
        {"server_args": {"max_total_tokens": 10000, "max_req_input_len": 10000}},
        {"max_req_input_len": True, "max_total_num_tokens": 3000},
        {"max_req_input_len": 2000, "max_total_num_tokens": "3000"},
    ],
)
def test_missing_or_option_capacity_is_unknown_not_filled_from_context(raw):
    result = measured_capacity(raw)
    assert result["max_req_input_len"] is None or result["max_total_num_tokens"] is None
    with pytest.raises(ValueError):
        require_capacity(result, input_tokens=2000, output_tokens=1000)


@pytest.mark.parametrize(
    ("input_cap", "total_cap", "valid"),
    [
        (1999, 5000, False),
        (2000, 2999, False),
        (0, 5000, False),
        (2000, 3000, True),
        (4096, 8192, True),
    ],
)
def test_input_and_full_request_fit_are_both_required(input_cap, total_cap, valid):
    result = measured_capacity({"max_req_input_len": input_cap, "max_total_num_tokens": total_cap})
    if valid:
        require_capacity(result, input_tokens=2000, output_tokens=1000)
    else:
        with pytest.raises(ValueError):
            require_capacity(result, input_tokens=2000, output_tokens=1000)


def test_explicit_worker_measurements_use_all_workers_and_do_not_guess_missing():
    workers = [
        {"max_req_input_len": 4000, "max_total_num_tokens": 8000},
        {"max_req_input_len": 3000, "max_total_num_tokens": 6000},
    ]
    result = measured_capacity({"internal_states": workers})
    assert result["max_req_input_len"] == 3000
    assert result["max_total_num_tokens"] == 6000
    require_capacity(result, input_tokens=2000, output_tokens=1000)
    workers[1].pop("max_req_input_len")
    result = measured_capacity({"internal_states": workers})
    assert result["max_req_input_len"] is None
    assert result["max_total_num_tokens"] == 6000
    result = measured_capacity({"max_req_input_len": 2000, "internal_states": workers})
    assert result["max_req_input_len"] == 2000
    assert result["sources"]["max_req_input_len"] == "server_info.top-level"
    assert (
        measured_capacity({"internal_states": [{"server_args": workers[0]}]})[
            "max_total_num_tokens"
        ]
        is None
    )
