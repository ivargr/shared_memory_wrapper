import logging
logging.basicConfig(level=logging.INFO)
from shared_memory_wrapper.shared_array_map_reduce import additative_shared_array_map_reduce
import numpy as np
import time


def func(a, b, c, chunk):
    time.sleep(0.1)
    return np.zeros(10) + b + chunk


def test_addititative():
    data = ("Hi", 1, 5)
    mapper = range(50)
    result_array = np.zeros(10)

    results = additative_shared_array_map_reduce(
        func, mapper, result_array, data, n_threads=4
    )


