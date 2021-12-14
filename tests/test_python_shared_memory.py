import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from shared_memory_wrapper.python_shared_memory import np_array_to_shared_memory, np_array_from_shared_memory, free_memory

def test():
    a = np.random.randint(0, 100, 10, dtype=np.uint64)
    np_array_to_shared_memory("test1", a)

    b = np_array_from_shared_memory("test1")
    assert np.all(a == b)

    c = np_array_from_shared_memory("test1")

    c[0] = 1000
    print(b[0])
    assert b[0] == 1000

test()

free_memory()