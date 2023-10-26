import multiprocessing
import logging
import time
import numpy as np
import os

import shared_memory_wrapper.shared_memory
from .shared_memory import SHARED_MEMORIES_IN_SESSION
from .shared_memory_v2 import object_to_shared_memory, object_from_shared_memory
from .shared_memory import remove_shared_memory
from multiprocessing import Pool, Queue, Process
import queue
from .util import chunker

"""
Module that implements a simpel map reduce method
where each process re-uses the same shared arrays for efficiency
"""



def _add_func(result_array, job_result):
    result_array += job_result


class FunctionWrapper:
    def __init__(self, function, data_id, add_func=_add_func, backend="shared_array"):
        self.function = function
        self.data_id = data_id
        self._backend = backend
        self._add_func = add_func

    def __call__(self, chunk_queue, result_array):
        logging.debug("Starting %s" % os.getpid())
        data = object_from_shared_memory(self.data_id, backend=self._backend)
        result_array = object_from_shared_memory(result_array)

        n_waited = 0
        t_prev_job = time.perf_counter()
        while True:
            try:
                run_specific_data = chunk_queue.get(block=True)
            except queue.Empty:
                if n_waited > 10:
                    logging.debug("Process waiting for queue (%d). Too many processes with too little to do?" % n_waited)
                n_waited += 1
                time.sleep(0.1)
                continue

            if run_specific_data is None:
                logging.debug("Run data is None, stopping process")
                return

            n_waited = 0
            t = time.perf_counter()
            logging.debug("Starting job")
            job_result = self.function(*data, run_specific_data)
            logging.debug("Result from job: %s" % job_result)
            self._add_func(result_array, job_result)
            logging.debug("Total time job took was %.3f, of which %.3f was actual job time" % (time.perf_counter()-t_prev_job, time.perf_counter()-t))
            t_prev_job = time.perf_counter()


def additative_shared_array_map_reduce(func, mapper, result_array, shared_data, n_threads=4, queue_size_factor=0.5, add_func=_add_func):
    #pool = get_shared_pool()
    shared_queue = Queue()
    shared_data_id = object_to_shared_memory(shared_data)

    # make the processes
    # make a new result array for each process
    t0 = time.perf_counter()
    result_arrays = [
        object_to_shared_memory(np.zeros_like(result_array)) for _ in range(n_threads)
    ]
    #functions = FunctionWrapper(func, shared_data_id)
    processes = [Process(target=FunctionWrapper(func, shared_data_id, add_func=add_func), args=(shared_queue,result_array)) for result_array in result_arrays]

    # start all processes
    for process in processes:
        process.start()

    max_queue = n_threads * queue_size_factor

    # feed chunks to the queue
    t = time.perf_counter()
    for chunk in mapper:
        t_queue_start = time.perf_counter()
        shared_queue.put(chunk)
        if time.perf_counter() - t_queue_start > 0.1:
            logging.warning("Putting in queue took %.3f" % (time.perf_counter() - t_queue_start))
        #logging.info("Time spent inserting job into queue: %.3f" % (time.perf_counter()-t))
        t = time.perf_counter()

        # don't make queue too big
        while shared_queue.qsize() > max_queue:
            logging.debug("Waiting to get more elements since queue is quite full. Max queue size is %d" % max_queue)
            logging.debug("Approx queue size now is %d" % shared_queue.qsize())
            time.sleep(0.1)

        #logging.info(f"{len(SHARED_MEMORIES_IN_SESSION)} shared memories in session now")

    # feed some None at the end to tell processes to stop waiting
    for _ in range(n_threads):
        shared_queue.put(None)

    # stop all the processes
    for process in processes:
        process.join()

    # collect results from each process
    t = time.perf_counter()
    for result in result_arrays:
        job_result = object_from_shared_memory(result)
        remove_shared_memory(result)
        #result_array = add_func(result_array, job_result)
        add_func(result_array, job_result)
    logging.debug("Time spent adding results in the end: %.3f" % (time.perf_counter()-t))

    remove_shared_memory(shared_data_id)

    return result_array