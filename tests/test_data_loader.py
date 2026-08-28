from collections import Counter
import math

import numpy as np
import pytest
import torch

from src.data_loader import get_batch


def test_get_batch_matches_cs336_contract():
    dataset = np.arange(100)
    batch_size = 32
    context_length = 7
    starting_indices: Counter[int] = Counter()

    for _ in range(1000):
        inputs, targets = get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device="cpu",
        )

        assert inputs.shape == (batch_size, context_length)
        assert targets.shape == (batch_size, context_length)
        assert inputs.dtype == torch.long
        assert targets.dtype == torch.long
        torch.testing.assert_close(targets, inputs + 1)

        starting_indices.update(inputs[:, 0].tolist())

    last_valid_start = len(dataset) - context_length - 1
    assert min(starting_indices) == 0
    assert max(starting_indices) == last_valid_start

    num_possible_starts = len(dataset) - context_length
    num_samples = 1000 * batch_size
    expected_count = num_samples / num_possible_starts
    standard_deviation = math.sqrt(
        num_samples
        * (1 / num_possible_starts)
        * (1 - 1 / num_possible_starts)
    )

    for count in starting_indices.values():
        assert expected_count - 5 * standard_deviation <= count
        assert count <= expected_count + 5 * standard_deviation


def test_get_batch_uses_requested_device():
    dataset = np.arange(100)

    with pytest.raises((RuntimeError, AssertionError)):
        get_batch(
            dataset=dataset,
            batch_size=4,
            context_length=7,
            device="cuda:99",
        )
