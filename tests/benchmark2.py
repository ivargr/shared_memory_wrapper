import logging
import time

logging.basicConfig(level=logging.DEBUG)
import numpy as np
from shared_memory_wrapper.shared_memory import run_numpy_based_function_in_parallel
from shared_memory_wrapper import free_memory


def func(b):
    return b + 100


a = np.arange(100000000)
t = time.perf_counter()

res = run_numpy_based_function_in_parallel(func, 8, [a])
print(res)
print(time.perf_counter()-t)
