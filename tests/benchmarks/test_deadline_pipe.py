from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator

import pytest
from skillev_private.benchmarks.deadline_pipe import DeadlinePipe


@pytest.fixture
def pipe() -> Iterator[tuple[DeadlinePipe, int, int]]:
    request_reader, request_writer = os.pipe()
    response_reader, response_writer = os.pipe()
    try:
        yield DeadlinePipe(request_writer, response_reader, 128), request_reader, response_writer
    finally:
        for fd in (request_reader, request_writer, response_reader, response_writer):
            os.close(fd)


@pytest.mark.parametrize("prefix", [b"", b'{"partial":'])
def test_entire_line_has_a_deadline(pipe: tuple[DeadlinePipe, int, int], prefix: bytes) -> None:
    channel, _, writer = pipe
    if prefix:
        os.write(writer, prefix)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        channel.read(deadline=started + 0.05)
    assert time.monotonic() - started < 0.5


def test_trickling_does_not_renew_deadline(pipe: tuple[DeadlinePipe, int, int]) -> None:
    channel, _, writer = pipe
    stop = threading.Event()

    def trickle() -> None:
        while not stop.wait(0.01):
            os.write(writer, b" ")

    thread = threading.Thread(target=trickle)
    thread.start()
    try:
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            channel.read(deadline=started + 0.06)
        assert time.monotonic() - started < 0.5
    finally:
        stop.set()
        thread.join()


def test_chunked_read_retains_following_response(pipe: tuple[DeadlinePipe, int, int]) -> None:
    channel, reader, writer = pipe
    deadline = time.monotonic() + 1
    channel.write(b'{"request":1}', deadline=deadline)
    assert os.read(reader, 128) == b'{"request":1}\n'
    os.write(writer, b'{"one":1}\n{"two":2}\n')
    assert channel.read(deadline=deadline) == b'{"one":1}'
    assert channel.read(deadline=deadline) == b'{"two":2}'


def test_oversized_response_fails(pipe: tuple[DeadlinePipe, int, int]) -> None:
    channel, _, writer = pipe
    os.write(writer, b"x" * 129 + b"\n")
    with pytest.raises(OSError):
        channel.read(deadline=time.monotonic() + 1)


def test_write_backpressure_uses_same_deadline(pipe: tuple[DeadlinePipe, int, int]) -> None:
    channel, _, _ = pipe
    while True:
        try:
            os.write(channel.request_fd, b"x" * 4096)
        except BlockingIOError:
            break
    with pytest.raises(TimeoutError):
        channel.write(b"request", deadline=time.monotonic() + 0.05)


def test_eof_after_partial_reply_is_not_a_response() -> None:
    reader, writer = os.pipe()
    input_reader, input_writer = os.pipe()
    try:
        channel = DeadlinePipe(input_writer, reader, 128)
        os.write(writer, b'{"incomplete":')
        os.close(writer)
        with pytest.raises(BrokenPipeError):
            channel.read(deadline=time.monotonic() + 1)
    finally:
        for fd in (reader, input_reader, input_writer):
            os.close(fd)
