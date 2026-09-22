from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from skillev.runtime.request_journal import DurableRequestJournal, UnknownRequestOutcomeError


def request(journal, send, **extra):
    values = {
        "identity": ("run-batch-trajectory", "1", "action", "policy1", "decoding1"),
        "endpoint": "http://actor-a/generate",
        "payload": {"input_ids": [1, 2, 3]},
        "send": send,
    }
    values.update(extra)
    return journal.request(**values)


def test_fresh_journal_object_restores_exact_response_without_send(tmp_path):
    path = tmp_path / "requests.sqlite3"
    status, raw = request(DurableRequestJournal(path), lambda: (200, {"output_ids": [4, 5]}))
    assert status == 200
    assert raw["output_ids"] == [4, 5]

    def forbidden():
        pytest.fail("completed response must not be resampled")

    status, restored = request(DurableRequestJournal(path), forbidden)
    assert status == 200
    assert restored["output_ids"] == [4, 5]
    assert restored["skillev_restored_response"] is True


def test_lost_response_is_unknown_and_never_implicitly_retried(tmp_path):
    path = tmp_path / "requests.sqlite3"
    calls = []

    def timeout():
        calls.append(1)
        raise TimeoutError("response lost")

    with pytest.raises(TimeoutError):
        request(DurableRequestJournal(path), timeout)
    with pytest.raises(UnknownRequestOutcomeError):
        request(DurableRequestJournal(path), timeout)
    assert calls == [1]


def test_partial_request_does_not_lock_other_requests_or_allow_duplicate_dispatch(tmp_path):
    journal = DurableRequestJournal(tmp_path / "requests.sqlite3")
    entered, release = Event(), Event()

    def send():
        entered.set()
        assert release.wait(5)
        return 200, {"output_ids": [5]}

    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(request, journal, send)
        assert entered.wait(5)
        try:
            with pytest.raises(UnknownRequestOutcomeError):
                request(journal, send)
            assert request(journal, lambda: (200, [9]), identity=("another-trajectory",))[1] == [9]
        finally:
            release.set()
        assert future.result()[0] == 200


@pytest.mark.parametrize(
    "extra",
    [
        {"payload": {"input_ids": [9]}},
        {"endpoint": "http://actor-b/generate"},
    ],
)
def test_same_coordinate_cannot_change_input_or_route(tmp_path, extra):
    journal = DurableRequestJournal(tmp_path / "requests.sqlite3")
    request(journal, lambda: (200, {}))
    with pytest.raises(ValueError):
        request(journal, lambda: (200, {}), **extra)


def test_complete_http_failure_is_not_a_successful_label_or_new_sample(tmp_path):
    journal = DurableRequestJournal(tmp_path / "requests.sqlite3")
    request(journal, lambda: (503, {"error": "unavailable"}))
    assert request(journal, lambda: pytest.fail("no retry"))[0] == 503


def test_process_exit_after_dispatch_does_not_allow_resampling(tmp_path):
    import subprocess
    import sys

    path = tmp_path / "crashed.sqlite3"
    code = """
import os, sys
from pathlib import Path
from skillev.runtime.request_journal import DurableRequestJournal
j = DurableRequestJournal(Path(sys.argv[1]))
j.request(identity=("crashed",), endpoint="http://actor/generate",
          payload={"input_ids":[1]}, send=lambda: os._exit(7))
"""
    result = subprocess.run(  # noqa: S603 - fixed owned crash fixture
        [sys.executable, "-c", code, str(path)], timeout=15, check=False
    )
    assert result.returncode == 7
    with pytest.raises(UnknownRequestOutcomeError):
        DurableRequestJournal(path).request(
            identity=("crashed",),
            endpoint="http://actor/generate",
            payload={"input_ids": [1]},
            send=lambda: pytest.fail("unknown dispatch cannot be replaced"),
        )


def test_episode_route_survives_new_instance_without_reassignment(tmp_path):
    path = tmp_path / "route.sqlite3"
    DurableRequestJournal(path).save_episode_route("episode", "policy", "http://b")
    journal = DurableRequestJournal(path)
    assert journal.episode_route("episode", "policy") == "http://b"
    with pytest.raises(ValueError):
        journal.save_episode_route("episode", "policy", "http://a")
    with pytest.raises(ValueError):
        journal.episode_route("episode", "other-policy")


def test_private_permissions_apply_before_wal_creation(tmp_path):
    import sqlite3
    import stat

    path = tmp_path / "requests.sqlite3"
    journal = DurableRequestJournal(path)
    # Keep a live connection so SQLite cannot delete its WAL before inspecting it.
    db = sqlite3.connect(path)
    try:
        db.execute("SELECT * FROM requests").fetchall()
        request(journal, lambda: (200, {"private_response": "original"}))
        for file in (path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")):
            assert stat.S_IMODE(file.stat().st_mode) == 0o600
    finally:
        db.close()


@pytest.mark.parametrize("retry_fails", [False, True])
def test_owner_authorization_is_exact_single_use_and_preserves_abort(tmp_path, retry_fails):
    import sqlite3

    path = tmp_path / "requests.sqlite3"
    journal = DurableRequestJournal(path)
    identity = ("run-batch-trajectory", "1", "action", "policy1", "decoding1")

    def timeout():
        raise TimeoutError("confirmed aborted request")

    with pytest.raises(TimeoutError):
        request(journal, timeout)
    grant = {
        "identity": identity,
        "authorization_id": "owner-one-retry",
        "reason": "server abort seen",
    }
    journal.authorize_aborted_retry(**grant)
    journal.authorize_aborted_retry(**grant)
    with pytest.raises(ValueError):
        request(journal, timeout, payload={"input_ids": [99]})
    with pytest.raises(ValueError):
        journal.authorize_aborted_retry(**{**grant, "authorization_id": "replacement"})
    if retry_fails:
        with pytest.raises(TimeoutError):
            request(journal, timeout)
        journal.authorize_aborted_retry(**grant)
        with pytest.raises(UnknownRequestOutcomeError):
            request(journal, lambda: pytest.fail("permission is already consumed"))
    else:
        assert request(journal, lambda: (200, {"output_ids": [8, 9]}))[1]["output_ids"] == [8, 9]
        assert request(journal, lambda: pytest.fail("do not sample again"))[1]["output_ids"] == [
            8,
            9,
        ]
    with sqlite3.connect(path) as db:
        before, after = db.execute(
            "SELECT prior_state,retry_state FROM authorized_retries"
        ).fetchone()
        assert before == "DISPATCHED"
        assert after == ("DISPATCHED" if retry_fails else "COMPLETED")


def test_retry_permission_cannot_apply_to_missing_or_completed_request(tmp_path):
    journal = DurableRequestJournal(tmp_path / "requests.sqlite3")
    grant = {"identity": ("completed",), "authorization_id": "owner", "reason": "abort"}
    with pytest.raises(ValueError):
        journal.authorize_aborted_retry(**grant)
    request(journal, lambda: (200, {}), identity=("completed",))
    with pytest.raises(ValueError):
        journal.authorize_aborted_retry(**grant)
