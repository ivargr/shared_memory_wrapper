import logging
import multiprocessing
from .shared_memory import object_to_shared_memory, object_from_shared_memory
from collections.abc import Iterable
from .shared_memory import get_shared_pool, close_shared_pool
import resource
import tracemalloc
import numpy as np


def interval_chunks(start, end, n_chunks):
    assert end > start
    assert n_chunks > 0
    boundaries = list(range(start, end, n_chunks))
    boundaries.append(end)
    return [(start, end) for start, end in zip(boundaries[0:-1], boundaries[1:])]


class Mapper:
    def start_next_job(self):
        pass

    def __iter__(self):
        raise NotImplementedError()


class RangeMapper(Mapper):
    def __init__(self, data, function, start, end):
        self.data = data
        self.function = function
        self.start = start
        self.end = end

    def __iter__(self):
        pass

    def run(self):
        # get data from shared memory, run on function with next start and end
        pass


class Reducer:
    # deals with the results from a job
    def add_result(self, result):
        raise NotImplementedError()

    def get_final_result(self):
        raise NotImplementedError()

    def __call__(self):
        return get_final_result()


class SumReducer(Reducer):
    def __init__(self):
        self.results = []

    def add_result(self, result):
        self.results.append(result)

    def get_final_result(self):
        return sum(self.results)


class NoReturnReducer(Reducer):
    def add_result(self):
        return

    def get_final_result(self):
        return None


class AddReducer(Reducer):
    def __init__(self, initial):
        self.result = initial

    def add_result(self, result):
        self.result += result

    def get_final_result(self):
        return self.result


class ConcatenateReducer(Reducer):
    def __init__(self, axis=0):
        self._axis = axis
        self.results = []

    def add_result(self, result):
        self.results.append(result)

    def get_final_result(self):
        return np.concatenate(self.results, axis=self._axis)


"""
def run_in_parallel(mapper, reducer, n_threads=8, reduce="element-wise-sum"):
    pool = multiprocessing.Pool(n_threads)
    for result in pool.imap(mapper.run, mapper):
        reducer.add_result(result)

    return reducer.get_final_result()
"""

class FunctionWrapper:
    def __init__(self, function, data_id, backend=None):
        self.function = function
        self.data_id = data_id
        self._backend = backend

    def __call__(self, run_specific_variable):
        log_memory_usage("Getting data from shared memory")
        data = object_from_shared_memory(self.data_id, backend=self._backend)
        log_memory_usage("Done getting data from shared memory")
        result = self.function(*data, run_specific_variable)
        logging.info("Done with job. Returning result")

        return result


def parallel_map_reduce_with_adding(function, data, mapper, initial_data, n_threads=8):
    reducer = AddReducer(initial_data)
    return parallel_map_reduce(function, data, mapper, reducer, n_threads)


def parallel_map_reduce(function, data, mapper, reducer=None, n_threads=8, backend="shared_array"):
    assert reducer is None or isinstance(reducer, Reducer)
    assert isinstance(mapper, Iterable), "Mapper must be iterable"
    assert isinstance(data, tuple)

    logging.info("Putting data in shared memory")
    data = object_to_shared_memory(data, backend=backend)
    logging.info("Done putting data in shared memory")
    pool = get_shared_pool(n_threads)

    function = FunctionWrapper(function, data, backend=backend)

    for result in pool.imap(function, mapper):
        logging.info("Result returned")
        if reducer is not None:
            assert result is not None
            reducer.add_result(result)

    close_shared_pool()

    if reducer is not None:
        return reducer.get_final_result()
    else:
        data = object_from_shared_memory(data, backend=backend)
        return data


def log_memory_usage(logplace=""):
    memory = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1000000
    logging.info("Memory usage (%s): %.4f GB" % (logplace, memory))
