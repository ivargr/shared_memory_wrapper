import logging
import numpy as np
from multiprocessing import shared_memory
import pickle
import os

SHARED_MEMORIES = []

def np_array_to_shared_memory(name, array):
    global SHARED_MEMORIES
    shm = shared_memory.SharedMemory(create=True, size=array.nbytes, name=name)
    holder_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
    holder_array[:] = array[:]  # Copy the original data into shared memory
    SHARED_MEMORIES.append(shm)

    # write metadata to a file so that we know the shape and dtype when
    # reading from shared memory later
    meta_data = (array.dtype, array.shape)
    file_name = "." + name + ".shm"
    with open(file_name, "wb") as f:
        pickle.dump(meta_data, f)


def np_array_from_shared_memory(name):
    global SHARED_MEMORIES
    with open("." + name + ".shm", "rb") as f:
        dtype, shape = pickle.load(f)
    existing_shm = shared_memory.SharedMemory(name=name)
    SHARED_MEMORIES.append(existing_shm)
    return np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)


def free_memory():
    global SHARED_MEMORIES
    for mem in SHARED_MEMORIES:
        try:
            mem.close()
            mem.unlink()
        except FileNotFoundError:
            continue
        os.remove("." + mem.name + ".shm")