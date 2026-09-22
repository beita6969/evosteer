import json

import pytest

from skillev.evolution.authoring_journal import UnresolvedAuthoringCallError
from skillev.evolution.authoring_transport import DurableAuthoringTransport


class Transport:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def request(self, **kwargs):
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def call(transport):
    return transport.request(
        method="POST",
        url="http://localhost/generate",
        payload={"input_ids": [1, 2]},
        timeout_seconds=1,
        max_response_bytes=1000,
    )


@pytest.mark.parametrize(
    "response",
    [(200, {"text": "malformed", "output_ids": [3, 4]}), (503, {"error": "unavailable"})],
)
def test_raw_response_survives_restart_without_a_second_request(tmp_path, response):
    path = tmp_path / "transport.json"
    first = Transport(response)
    assert call(DurableAuthoringTransport(first, path)) == response
    unused = Transport(RuntimeError("must not call network"))
    assert call(DurableAuthoringTransport(unused, path)) == response
    assert first.calls == 1
    assert unused.calls == 0
    assert json.loads(path.read_text())["response"] == response[1]


def test_ambiguous_network_failure_stays_fail_closed(tmp_path):
    path = tmp_path / "transport.json"
    first = Transport(TimeoutError("response lost"))
    with pytest.raises(TimeoutError):
        call(DurableAuthoringTransport(first, path))
    unused = Transport((200, {}))
    with pytest.raises(UnresolvedAuthoringCallError):
        call(DurableAuthoringTransport(unused, path))
    assert unused.calls == 0
