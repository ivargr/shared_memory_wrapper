import random
import pickle
import time
import inspect
import os
import SharedArray as sa
import logging
import numpy as np
from itertools import repeat
from pathos.multiprocessing import Pool
import inspect
#from . import python_shared_memory
from collections import OrderedDict
#from . import posix_shared_memory

SHARED_MEMORIES_IN_SESSION = []
TMP_FILES_IN_SESSION = []

class DataBundle:
    """"
    Simple wrapper around np.savez for wrapping multiple files in an archive
    """
    def __init__(self, file_name, backend="file"):
        assert backend in ["compressed_file", "file", "shared_array", "python", "posix"]
        self._file_name = file_name
        self._backend = backend
        self._description = {}
        self._data = {}

    def add(self, name, data):
        self._data[name] = data

    def save(self, description_dict):
        global TMP_FILES_IN_SESSION

        save_to_file = {}  # will save these to file in the end
        if self._backend == "file" or self._backend == "compressed_file":
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
        if self._backend == "compressed_file":
            logging.info("Saving to compressed file")
            np.savez_compressed(self._file_name, **save_to_file)
        else:
            np.savez(self._file_name, **save_to_file)

        if self._backend != "file" and self._backend != "compressed_file":
            TMP_FILES_IN_SESSION.append(self._file_name + ".npz")
        #logging.info("Saved to %s" % self._file_name)


class SingleSharedArray:
    properties = {"array"}
    def __init__(self, array=None):
        self.array = array


_shared_pool = None


def get_shared_pool(n_threads=16):
    global _shared_pool
    if _shared_pool is None:
        t = time.perf_counter()
        _shared_pool = Pool(n_threads)
        logging.info("Made shared pool, took %.4f sec" % (time.perf_counter()-t))

    return _shared_pool


def close_shared_pool():
    global _shared_pool
    if _shared_pool is not None:
        _shared_pool.close()
        _shared_pool = None
        logging.info("Closed shared pool")


def _get_class_init_arguments(cls):
    arguments = list(inspect.getfullargspec(cls.__init__))[0]
    arguments.remove("self")
    return arguments


def _get_object_init_arguments(object):
    arguments = list(inspect.getfullargspec(object.__class__.__init__))[0]
    arguments.remove("self")
    #arguments.append("kwargs")
    return arguments

def array_from_shared_memory(name, backend="shared_array"):
    if backend == "shared_array":
        return sa.attach(name)
    elif backend == "python":
        assert False, "Not supported"
        return python_shared_memory.np_array_from_shared_memory(name)
    elif backend == "posix":
        assert False, "Not supported"
        return posix_shared_memory.np_array_from_shared_memory(name)
    elif backend == "file":
        return np.load("." + name + ".npy")
    elif backend == "compressed_file":
        return np.load("." + name + ".npz")["arr_0"]
    else:
        raise Exception("Invalid backend %s" % backend)


def array_to_shared_memory(name, array, backend):
    if name is None:
        name = random_name()

    assert isinstance(name, str)
    global SHARED_MEMORIES_IN_SESSION
    if backend == "shared_array":
        try:
            sa.delete(name)
        except FileNotFoundError:
            pass

        shared_array = sa.create(name, array.shape, array.dtype)
        try:
            shared_array[:] = array
        except IndexError:
            logging.error("Error when trying to save %s to shared memory" % name)
            raise
        SHARED_MEMORIES_IN_SESSION.append(name)
    elif backend == "python":
        assert False, "Not supported"
        python_shared_memory.np_array_to_shared_memory(name, array)
    elif backend == "posix":
        assert False, "Not supported"
        posix_shared_memory.np_array_to_shared_memory(name, array)
    elif backend == "file":
        np.save("." + name + ".npy", array, allow_pickle=True)
    elif backend == "compressed_file":
        np.savez_compressed("." + name + ".npz", array, allow_pickle=True)
    else:
        raise Exception("Invalid backend %s" % backend)

    return name


def base_type_to_shared_memory(object, name):
    # base types (strings, ints etc) are just pickled for simplicity
    with open("." + name + ".shm", "wb") as f:
        pickle.dump(object, f)


def base_type_from_shared_memory(name):
    with open("." + name + ".shm", "rb") as f:
        return pickle.load(f)


def from_file(name):
    return object_from_shared_memory(name, "file")


def to_file(object, base_name=None, compress=False):
    if base_name is not None and base_name.endswith(".npz"):
        base_name = base_name.replace(".npz", "")

    backend = "file" if not compress else "compressed_file"
    return object_to_shared_memory(object, base_name, backend=backend)


def random_name():
    random_generator = random.Random()  # create new generator so seed does not affect
    return str(random_generator.randint(0, 10e15))


def object_to_shared_memory(object, base_name=None, backend="shared_array"):
    t = time.perf_counter()
    if base_name is None:
        base_name = random_name()

    data_bundle = DataBundle(base_name, backend)

    if "/" in base_name:
        logging.info("Base name is a file path: %s" % base_name)
        base_name = base_name.split("/")[-1]
        logging.info("Using last par of file path as base name: %s" % base_name)

    description = _object_to_shared_memory(object, base_name, data_bundle, backend)
    #description = (object.__class__, description)
    data_bundle.save(description)

    #with open("." + base_name + ".shm", "wb") as f:
    #    pickle.dump(description, f)

    return base_name


def _object_is_basetype(object):
    if isinstance(object, int) or isinstance(object, str) or isinstance(object, float):
        return True
    elif isinstance(object, np.integer):
        return True
    return False


def _is_iterable(object):
    try:
        iter(object)
        return True
    except TypeError:
        return False


def _object_to_shared_memory(object, name, data_bundle, backend="shared_array"):
    """
    Function will be called recursively. Responsibility: Put object in shared memory,
    and return tuple of (object description, description dict from calling same method
    on children if object has children)
    """

    if _object_is_basetype(object):
        data_bundle.add(name, object)
        return ("pickle", None)
    elif isinstance(object, np.ndarray):
        data_bundle.add(name, object)
        return ("ndarray", None)
    elif isinstance(object, set):
        return (object.__class__, array_to_shared_memory(name, np.array([e for e in object]), backend))
    elif issubclass(list, object.__class__) or issubclass(tuple, object.__class__):
        return (object.__class__, [_object_to_shared_memory(element, name + "-" + str(i), data_bundle, backend)
                         for i, element in enumerate(object)])
    elif issubclass(dict, object.__class__):
        return (object.__class__, {key: _object_to_shared_memory(value, name + "-" + str(key), data_bundle, backend)
                                   for key, value in object.items()})
    elif _is_iterable(object):
        # if iterable and non of above, assume we can wrap in a tuple
        return (object.__class__, tuple(_object_to_shared_memory(element, name + "-" + str(i), data_bundle, backend)
                                   for i, element in enumerate(object)))
        #return (object.__class__, _object_to_shared_memory(tuple(object), name + "-" + str(object.__class__.__name__),
        #                                                   data_bundle, backend))
    else:
        # is an object with possibly children
        # go through children and put these in shared memory
        variable_names = _get_object_init_arguments(object)
        #logging.info("Variable names in %s: %s" % (object.__class__, variable_names))
        description = OrderedDict()
        for variable_name in variable_names:
            if not hasattr(object, variable_name):
                variable_name = "_" + variable_name
                if not (hasattr(object, variable_name)):
                    # last attempt: Try finding private variables
                    for possible_var in object.__dict__.keys():
                        if "__" in possible_var:
                            if possible_var.split("__")[1] == variable_name[1:]:
                                variable_name = possible_var
                                break
                    if not (hasattr(object, variable_name)):
                        logging.warning("Object %s has init argument %s, but no property with the same name. Ignoring" % (
                        object, variable_name))
                        continue

            variable_data = getattr(object, variable_name)
            child_shared_memory_name = name + "-" + variable_name
            description[variable_name] = _object_to_shared_memory(variable_data, child_shared_memory_name, data_bundle, backend)

        return (object.__class__, description)


def _list_to_shared_memory(object, name, data_bundle, backend):
    return (list, OrderedDict({i: _object_to_shared_memory(element, name + "-" + str(i), data_bundle, backend)
                               for i, element in enumerate(object)}))


def object_from_shared_memory(name, backend="shared_array"):
    try:
        data_bundle = np.load(name + ".npz", allow_pickle=True)
    except FileNotFoundError:
        try:
            data_bundle = np.load(name, allow_pickle=True)
            name = name.replace(".npz", "")
        except FileNotFoundError:
            logging.error("Object file not found")
            raise

    description = data_bundle["__description"]
    if "/" in name:
        logging.info("Base name is a file path: %s" % name)
        name = name.split("/")[-1]
        logging.info("Using last par of file path as base name: %s" % name)

    return _object_from_shared_memory(name, description, data_bundle, backend)


def _object_from_shared_memory(name, description, data_bundle, backend="shared_array"):
    object_type, children = description

    if children is None:
        # no children
        if object_type == "pickle":
            return np.atleast_1d(data_bundle[name])[0]
        elif object_type == "ndarray":
            if backend == "file" or backend == "compressed_file":
                return data_bundle[name]
            else:
                return array_from_shared_memory(name, backend)
        elif issubclass(set, object_type):
            return set(array_from_shared_memory(name, backend))
        else:
            raise Exception("Error: %s" % description)
    else:
        # has children
        if issubclass(list, object_type) or issubclass(tuple, object_type):
            #print("      TUple with children: %s" % children)
            return object_type([_object_from_shared_memory(name + "-" + str(i), e, data_bundle, backend)
                    for i, e in enumerate(children)])
        elif issubclass(dict, object_type):
            return object_type((key_name, _object_from_shared_memory(name + "-" + str(key_name), child_desc, data_bundle, backend))
                                for key_name, child_desc in children.items())
        else:
            data = []
            for child_name, child_description in children.items():
                shared_memory_name = name + "-" + child_name
                data.append(_object_from_shared_memory(shared_memory_name, child_description, data_bundle, backend))

            return object_type(*data)


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
        if type(data) == int or data.shape == ():
            if type(data) == int:
                dtype=int
            else:
                dtype=data.dtype
            data = np.array([data], dtype=dtype)

        array_to_shared_memory(shared_memory_name, data, "python" if use_python_backend else "shared_array")

    return name


def remove_shared_memory(name, limit_to_session=False):
    if limit_to_session:
        shared_memories = SHARED_MEMORIES_IN_SESSION
    else:
        shared_memories = [s.name.decode("utf-8") for s in sa.list()]


    n_deleted = 0
    for m in shared_memories:
        if m.startswith(name + "__") or m.startswith(name + "-") or m == name:
            try:
                sa.delete(m)
                n_deleted += 1
            except FileNotFoundError:
                logging.debug("Did not delete %s" % m)
                continue
            if m in SHARED_MEMORIES_IN_SESSION:
                SHARED_MEMORIES_IN_SESSION.remove(m)

    # if object contained no numpy arrays, no shared memory"
    # was needed, but we may still have a file
    if "__" not in name:
        # not a subname
        if name in SHARED_MEMORIES_IN_SESSION:
            SHARED_MEMORIES_IN_SESSION.remove(name)
        file = name + ".npz"
        if os.path.exists(file):
            os.remove(file)

        if name in TMP_FILES_IN_SESSION:
            TMP_FILES_IN_SESSION.remove(name)

        n_deleted += 1

    if n_deleted == 0:
        logging.debug("Did not find anything to delete fro %s" % name)


def remove_shared_memory_in_session():
    for name in SHARED_MEMORIES_IN_SESSION:
        remove_shared_memory(name, True)


def remove_all_shared_memory():
    global SHARED_MEMORIES_IN_SESSION
    for shared in sa.list():
        name = shared.name.decode("utf-8")
        sa.delete(name)
        if name in SHARED_MEMORIES_IN_SESSION:
            SHARED_MEMORIES_IN_SESSION.remove(name)


def free_memory():
    remove_all_shared_memory()


def free_memory_in_session():
    remove_shared_memory_in_session()

def _run_numpy_based_function_on_shared_memory_arguments(function, input_arguments, interval):
    start_time = time.perf_counter()
    start, end = interval
    #logging.info("Running on interval %d-%d" % (start, end))
    arguments = []
    for a in input_arguments:
        if isinstance(a, str):
            try:
                object = from_shared_memory(SingleSharedArray, a).array
                arguments.append(object)
            except FileNotFoundError:
                logging.debug("Did not find shared memory %s. Assuming this is a real string and not a shared memory name" % a)
                arguments.append(a)
        else:
            arguments.append(a)

    sliced_arguments = []
    for argument in arguments:
        if isinstance(argument, np.ndarray) and argument.shape[0] != 1:
            argument = argument[start:end]
        sliced_arguments.append(argument)

    result = function(*sliced_arguments)
    shared_memory_name = str(random.randint(0,10e15))
    to_shared_memory(SingleSharedArray(result), shared_memory_name)
    #logging.info("Interval %d-%d took %.4f sec" % (start, end, time.perf_counter()-start_time))
    return shared_memory_name


def run_numpy_based_function_in_parallel(function, n_threads, arguments, chunks=None):
    # split every argument that is not an int
    # put each in shared memory
    # run function on chunk on separete thread, put in result array
    # return result array
    if n_threads == 1:
        logging.info("n_threads is 1. Not running in paralell")
        return function(*arguments)

    new_arguments = []

    # Put np arrays in shared memory, everything else we keep as is
    array_length = 0
    for i, argument in enumerate(arguments):
        # don't split ndarrays with shape[0] == 1
        if isinstance(argument, np.ndarray) and argument.shape[0] != 1:
            argument_id = str(np.random.randint(0, 10e15))
            t0 = time.perf_counter()
            to_shared_memory(SingleSharedArray(argument), argument_id)
            logging.info("Time to write to shared memory: %.3f" % (time.perf_counter()-t0))
            new_arguments.append(argument_id)
            if array_length != 0 and array_length != argument.shape[0]:
                logging.error("Found argument with different shape")
                logging.error("Shape: %s" % str(argument.shape))
                logging.error("ARray length: %d" % array_length)
                logging.error("Argument #%d" % i)
                logging.error("Data: %s" % argument)
                raise Exception("")

            array_length = argument.shape[0]

        else:
            new_arguments.append(argument)

    t = time.perf_counter()
    pool = get_shared_pool(n_threads)  # Pool(n_threads)

    if chunks is None:
        intervals = list([int(i) for i in np.linspace(0, array_length, n_threads + 1)])
        intervals = [(from_pos, to_pos) for from_pos, to_pos in zip(intervals[0:-1], intervals[1:])]
    else:
        logging.debug("Using predefined chunks")
        intervals = chunks

    logging.info("Will run on intervals %s" % intervals)

    results = []
    for result_name in pool.starmap(_run_numpy_based_function_on_shared_memory_arguments, zip(repeat(function), repeat(new_arguments), intervals)):
        result = from_shared_memory(SingleSharedArray, result_name).array
        results.append(result)
        sa.delete(result_name + "__array")

    # Free memory for arrays used
    for argument in new_arguments:
        if isinstance(argument, str):
            remove_shared_memory(argument, limit_to_session=True)

    t = time.perf_counter()
    results = np.concatenate(results)
    logging.debug("Time to concatenate results: %.3f" % (time.perf_counter()-t))
    return results


def np_array_to_shared_memory(array, name):
    to_shared_memory(SingleSharedArray(array), name)


def np_array_from_shared_memory(name):
    return from_shared_memory(SingleSharedArray, name).array

