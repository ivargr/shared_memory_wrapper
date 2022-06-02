import logging
import numpy as np
from multiprocessing import shared_memory
import pickle
import os

_shared_data = {}


def np_array_to_shared_memory(name, array):
    global _shared_data
    logging.info("ADDING TO POSIX ARRAY: %s" % name)
    _shared_data[name] = array


def np_array_from_shared_memory(name):
    global _shared_data
    try:
        return _shared_data[name]
    except KeyError:
        logging.error("Looked for %s" % name)
        logging.error("Possible data keys: %s" % list(_shared_data.keys()))
        raise


def free_memory():
    global _shared_data
    _shared_data = {}
