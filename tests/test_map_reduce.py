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


