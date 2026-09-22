"""One experiment seed, reproducible fresh randomness on every admitted call."""

from skillev.evaluation.sampling_stream import evaluation_call_seed


def test_sampled_call_stream_advances_and_reproduces_without_multiple_experiments():
    seed = 17
    stream = [evaluation_call_seed(seed, index, sampling_mode="sampling") for index in range(160)]
    assert stream[0] == seed
    assert len(set(stream)) == len(stream)
    assert stream == [
        evaluation_call_seed(seed, index, sampling_mode="sampling") for index in range(160)
    ]


def test_call_stream_stays_in_the_existing_unsigned_seed_domain():
    seed = (1 << 64) - 1
    assert evaluation_call_seed(seed, 0, sampling_mode="sampling") == seed
    assert evaluation_call_seed(seed, 1, sampling_mode="sampling") == 0


def test_greedy_call_keeps_its_original_seed():
    assert evaluation_call_seed(17, 159, sampling_mode="greedy") == 17
