import pytest
from shared_memory_wrapper.util import interval_chunks

def test_interval_chunks():
    assert interval_chunks(0, 3, 1) == [(0, 1), (1, 2), (2, 3)]
    assert interval_chunks(0, 10, 3) == [(0, 3), (3, 6), (6, 9), (9, 10)]
    assert interval_chunks(1, 2, 3) == [(1, 2)]

