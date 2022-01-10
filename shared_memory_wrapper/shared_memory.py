import random
import pickle
import time
import inspect

import SharedArray as sa
import logging
import numpy as np
from itertools import repeat
from pathos.multiprocessing import Pool
import inspect
from . import python_shared_memory
from collections import OrderedDict



class DataBundle:
    """"
    Simple wrapper around np.savez for wrapping multiple files in an archive
    """
    def __init__(self, file_name, backend="file"):
        assert backend in ["file", "shared_array", "python"]
        self._file_name = file_name
        self._backend = backend
        self._description = {}
        self._data = {}

    def add(self, name, data):
        self._data[name] = data

    def save(self, description_dict):

        save_to_file = {}  # will save these to file in the end
        if self._backend == "file":
            # everything can be saved to file
            save_to_file = self._data
        else:
            # np arrays to shared array, objects to file
            for name, data in self._data.items():
                if isinstance(data, np.ndarray):
                    array_to_shared_memory(name, data, self._backend)
                else:
                    #base_type_to_shared_memory(data, name)
                    save_to_file[name] = data

        save_to_file["__description"]  = description_dict
        np.savez(self._file_name, **save_to_file)
        logging.info("Saved to %s" % self._file_name)


class SingleSharedArray:
    properties = {"array"}
    def __init__(self, array=None):
        self.array = array

SHARED_MEMORIES_IN_SESSION = []

_shared_pool = None


def get_shared_pool(n_threads=16):
    global _shared_pool
    if _shared_pool is None:
        _shared_pool = Pool(n_threads)
        logging.info("Made shared pool")

    return _shared_pool


def close_shared_pool():
    global _shared_pool
    if _shared_pool is not None:
        _shared_pool.close()
        logging.info("Closed shared pool")


def _get_class_init_arguments(cls):
    arguments = list(inspect.getfullargspec(cls.__init__))[0]
    arguments.remove("self")
    return arguments


def _get_object_init_arguments(object):
    arguments = list(inspect.getfullargspec(object.__class__.__init__))[0]
    arguments.remove("self")
    return arguments

def array_from_shared_memory(name, backend="shared_array"):
    if backend == "shared_array":
        return sa.attach(name)
    elif backend == "python":
        return python_shared_memory.np_array_from_shared_memory(name)
    elif backend == "file":
        return np.load("." + name + ".npy")
    else:
        raise Exception("Invalid backend %s" % backend)


def array_to_shared_memory(name, array, backend):
    global SHARED_MEMORIES_IN_SESSION
    if backend == "shared_array":
        try:
            sa.delete(name)
        except FileNotFoundError:
            pass

        shared_array = sa.create(name, array.shape, array.dtype)
        shared_array[:] = array
        SHARED_MEMORIES_IN_SESSION.append(name)
    elif backend == "python":
        python_shared_memory.np_array_to_shared_memory(name, array)
    elif backend == "file":
        np.save("." + name + ".npy", array, allow_pickle=True)
    else:
        raise Exception("Invalid backend %s" % backend)



def base_type_to_shared_memory(object, name):
    # base types (strings, ints etc) are just pickled for simplicity
    with open("." + name + ".shm", "wb") as f:
        pickle.dump(object, f)


def base_type_from_shared_memory(name):
    with open("." + name + ".shm", "rb") as f:
        return pickle.load(f)


def from_file(name):
    return object_from_shared_memory(name, "file")


def to_file(object, base_name=False):
    return object_to_shared_memory(object, base_name, backend="file")


def object_to_shared_memory(object, base_name=None, backend="shared_array"):
    if base_name is None:
        random_generator = random.Random()  # create new generator so seed does not affect
        base_name = str(random_generator.randint(0, 10e15))

    data_bundle = DataBundle(base_name, backend)
    description = _object_to_shared_memory(object, base_name, data_bundle, backend)
    description = (object.__class__, description)
    data_bundle.save(description)

    #with open("." + base_name + ".shm", "wb") as f:
    #    pickle.dump(description, f)

    return base_name


def _object_to_shared_memory(object, name, data_bundle, backend="shared_array"):
    variable_names = _get_object_init_arguments(object)
    description = OrderedDict()
    for variable_name in variable_names:
        if not hasattr(object, variable_name):
            variable_name = "_" + variable_name
            if not(hasattr(object, variable_name)):
                logging.warning("Object %s has init argument %s, but no property with the same name. Ignoring" % (object, variable_name))
                continue

        variable_data = getattr(object, variable_name)
        shared_memory_name = name + "-" + variable_name

        if isinstance(variable_data, int) or isinstance(variable_data, str) or isinstance(variable_data, float):
            description[variable_name] = ("pickle", None)
            data_bundle.add(shared_memory_name, variable_data)
        elif isinstance(variable_data, np.ndarray):
            description[variable_name] = ("ndarray", None)
            data_bundle.add(shared_memory_name, variable_data)
        else:
            # try to save this object recursively to shared memory
            desc = _object_to_shared_memory(variable_data, shared_memory_name, data_bundle, backend)
            description[variable_name] = (variable_data.__class__, desc)

    return description


def object_from_shared_memory(name, backend="shared_array"):
    data_bundle = np.load(name + ".npz", allow_pickle=True)
    cls, description = data_bundle["__description"]
    return _object_from_shared_memory(name, cls, description, data_bundle, backend)


def _object_from_shared_memory(name, cls, description, data_bundle, backend="shared_array"):
    data = []
    for attribute, attribute_description in description.items():
        attribute_type = attribute_description[0]
        shared_memory_name = name + "-" + attribute
        if attribute_type == "pickle":
            #data.append(base_type_from_shared_memory(shared_memory_name))
            data.append(np.atleast_1d(data_bundle[shared_memory_name])[0])
        elif attribute_type == "ndarray":
            if backend == "file":
                data.append(data_bundle[shared_memory_name])
            else:
                data.append(array_from_shared_memory(shared_memory_name, backend))
        else:
            # attribute is object
            data.append(_object_from_shared_memory(shared_memory_name, attribute_type, attribute_description[1], data_bundle, backend))

    return cls(*data)


def from_shared_memory(cls, name, use_python_backend=False):
    property_names = _get_class_init_arguments(cls)

    init_data = []

    for property_name in property_names:
        data = array_from_shared_memory(name + "__" + property_name, "python" if use_python_backend else "shared_array")
        # Single ints are wrapped in arrays
        if len(data) == 1 and property_name == "_modulo":
            data = data[0]
        init_data.append(data)

    return cls(*init_data)


def to_shared_memory(object, name=None, use_python_backend=False):
    global SHARED_MEMORIES_IN_SESSION

    if name is None:
        random_generator = random.Random()  # create new generator so seed does not affect
        name = str(random_generator.randint(0,10e15))


    # write each instance variable to shared memory
    variables = _get_object_init_arguments(object)
    meta_information = {}
    for variable in variables:
        shared_memory_name = name + "__" + variable

        if not hasattr(object, variable):
            variable = "_" + variable
            if not(hasattr(object, variable)):
                raise Exception("Object %s has init argument %s, but no property with the same name" % (object, variable))

        data = getattr(object, variable)
        if data is None:
            data = np.zeros(0)

        # Wrap single ints in arrays
        if data.shape == ():
            data = np.array([data], dtype=data.dtype)

        array_to_shared_memory(shared_memory_name, data, "python" if use_python_backend else "shared_array")

    return name


def remove_shared_memory(name):
    shared_memories = [s.name.decode("utf-8") for s in sa.list()]

    for m in shared_memories:
        if m.startswith(name + "__"):
            sa.delete(m)
            return

    logging.warning("No shared memory with name %s" % name)
    logging.warning("Available shared memories: %s" % shared_memories)


def remove_shared_memory_in_session():
    for name in SHARED_MEMORIES_IN_SESSION:
        try:
            sa.delete(name)
        except FileNotFoundError:
            pass
            #logging.warning("Tried to deleted shared memory %s that did not exist" % name)

    python_shared_memory.free_memory()

def remove_all_shared_memory():
    for shared in sa.list():
        sa.delete(shared.name.decode("utf-8"))

def free_memory():
    remove_all_shared_memory()


def free_memory_in_session():
    remove_shared_memory_in_session()

def _run_numpy_based_function_on_shared_memory_arguments(function, arguments, interval):
    start_time = time.perf_counter()
    start, end = interval
    #logging.info("Running on interval %d-%d" % (start, end))
    arguments = [from_shared_memory(SingleSharedArray, a).array if type(a) == str else a for a in arguments]
    sliced_arguments = []
    for argument in arguments:
        if isinstance(argument, np.ndarray):
            argument = argument[start:end]
        sliced_arguments.append(argument)

    result = function(*sliced_arguments)
    shared_memory_name = str(random.randint(0,10e15))
    to_shared_memory(SingleSharedArray(result), shared_memory_name)
    #logging.info("Interval %d-%d took %.4f sec" % (start, end, time.perf_counter()-start_time))
    return shared_memory_name


def run_numpy_based_function_in_parallel(function, n_threads, arguments):
    # split every argument that is not an int
    # put each in shared memory
    # run function on chunk on separete thread, put in result array
    # return result array
    new_arguments = []

    # Put np arrays in shared memory, everything else we keep as is
    array_length = 0
    for argument in arguments:
        if isinstance(argument, np.ndarray):
            argument_id = str(np.random.randint(0, 10e15))
            to_shared_memory(SingleSharedArray(argument), argument_id)
            new_arguments.append(argument_id)
            assert array_length == 0 or array_length == argument.shape[0], "Found argument with different shape"
            array_length = argument.shape[0]

        else:
            new_arguments.append(argument)

    t = time.perf_counter()
    pool = get_shared_pool(n_threads)  # Pool(n_threads)

    intervals = list([int(i) for i in np.linspace(0, array_length, n_threads + 1)])
    intervals = [(from_pos, to_pos) for from_pos, to_pos in zip(intervals[0:-1], intervals[1:])]
    #logging.info("Will run on intervals %s" % intervals)

    results = []
    for result in pool.starmap(_run_numpy_based_function_on_shared_memory_arguments, zip(repeat(function), repeat(new_arguments), intervals)):
        result = from_shared_memory(SingleSharedArray, result).array
        results.append(result)

    t = time.perf_counter()
    results = np.concatenate(results)
    logging.debug("Time to concatenate results: %.3f" % (time.perf_counter()-t))
    return results


def np_array_to_shared_memory(array, name):
    to_shared_memory(SingleSharedArray(array), name)


def np_array_from_shared_memory(name):
    return from_shared_memory(SingleSharedArray, name).array

