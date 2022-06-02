import logging
logging.basicConfig(level=logging.INFO)
from shared_memory_wrapper.util import parallel_map_reduce, Reducer, Mapper
from shared_memory_wrapper.util import log_memory_usage, interval_chunks
from shared_memory_wrapper import get_shared_pool, close_shared_pool
import numpy as np
from shared_memory_wrapper.posix_shared_memory import np_array_to_shared_memory, np_array_from_shared_memory
np.random.seed(1)
from shared_memory_wrapper import free_memory_in_session
from shared_memory_wrapper import from_file

class MyReducer(Reducer):
    def __init__(self):
        self._results = 0

    def add_result(self, result):
        self._results += result

    def get_final_result(self):
        return self._results


class Wrapper:
    def __init__(self, data):
        self.data = data


def some_function(data, interval_range):
    log_memory_usage("Starting interval range %s" % str(interval_range))
    #out = np.sum(data) + interval_range[0]
    #for i in data:
    #    out += i
    log_memory_usage("Done range %s" % str(interval_range))
    #while True:
    #    continue
    data.count_kmers(data.kmers)
    out = 100
    return out

def test_posix():
    a = np.arange(1000)
    np_array_to_shared_memory("test1", a)
    b = np_array_from_shared_memory("test1")


    assert np.all(a == b)


def test():
    n_threads = 10
    get_shared_pool(n_threads)
    log_memory_usage("Starting")
    n = 1000000000 * 4
    #data = np.random.randint(0, 10, n, dtype=np.int8)
    data = from_file("../counter_index_only_variants_with_revcomp.npz")
    log_memory_usage("Data created")


    reducer = interval_chunks(0, 200, 10)
    print(list(reducer))
    results = parallel_map_reduce(some_function, (data,),
                                  reducer,
                                  MyReducer(),
                                  n_threads,
                                  backend="shared_array")


    print(results)
    log_memory_usage("Before closing shared pool")
    close_shared_pool()
    log_memory_usage("After closing shared pool")
    free_memory_in_session()



test()