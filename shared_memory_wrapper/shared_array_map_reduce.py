import multiprocessing
import logging
import time
import numpy as np

from .shared_memory import get_shared_pool, remove_shared_memory, object_to_shared_memory, object_from_shared_memory
from multiprocessing import Pool, Queue, Process
import queue
from .util import chunker

"""
Module that implements a simpel map reduce method
where each process re-uses the same shared arrays for efficiency
"""



class FunctionWrapper:
    def __init__(self, function, data_id, backend="shared_array"):
        self.function = function
        self.data_id = data_id
        self._backend = backend

    def __call__(self, chunk_queue, result_array):
        data = object_from_shared_memory(self.data_id, backend=self._backend)
        result_array = object_from_shared_memory(result_array)

        while True:
            try:
                run_specific_data = chunk_queue.get(block=False)
            except queue.Empty:
                continue

            if run_specific_data is None:
                return

            result_array += self.function(*data, run_specific_data)


def additative_shared_array_map_reduce(func, mapper, result_array, shared_data, n_threads=4):
    #pool = get_shared_pool()
    shared_queue = Queue()
    shared_data_id = object_to_shared_memory(shared_data)

    # make the processes
    # make a new result array for each process
    result_arrays = [
        object_to_shared_memory(np.zeros_like(result_array)) for _ in range(n_threads)
    ]
    functions = FunctionWrapper(func, shared_data_id)
    processes = [Process(target=FunctionWrapper(func, shared_data_id), args=(shared_queue,result_array)) for result_array in result_arrays]

    # start all processes
    for process in processes:
        process.start()

    # feed chunks to the queue
    for chunk in mapper:
        shared_queue.put(chunk)

        # don't make queue too big
        while shared_queue.qsize() > n_threads*3:
            #logging.info("Waiting to get more elements since queue is quite full")
            #logging.info("Approx queue size now is %d" % shared_queue.qsize())
            time.sleep(1)

    # feed some None at the end to tell processes to stop waiting
    for _ in range(n_threads):
        shared_queue.put(None)

    # stop all the processes
    for process in processes:
        process.join()

    # collect results from each process
    t = time.perf_counter()
    for result in result_arrays:
        result_array += object_from_shared_memory(result)
    logging.info("Time spent adding results in the end: %.3f" % (time.perf_counter()-t))


    return result_array