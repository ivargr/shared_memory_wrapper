from .object_traversal import replace_object_attributes_recursively
from .shared_memory import array_to_shared_memory, array_from_shared_memory, random_name
from .shared_memory import TMP_FILES_IN_SESSION, SHARED_MEMORIES_IN_SESSION
import numpy as np
import pickle
import logging


def pickle_object(object, name):
    with open(name, "wb") as f:
        pickle.dump(object, f, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_object(name):
    with open(name, "rb") as f:
        return pickle.load(f)


class ObjectWrapper:
    def __init__(self, object):
        self.object = object


class _Wrapper:
    def __init__(self, object):
        self.object = object


class DataBundle:
    def __init__(self, object, file_name, backend="shared_array"):
        self._file_name = file_name
        self._backend = backend
        self._object = object
        self._np_arrays = {}

    def save(self):
        if self._backend == "file" or self._backend == "compressed_file":
            # save object and np_arrays to file
            func = np.savez
            if self._backend == "compressed_file":
                func = np.savez_compressed
            d = {**{"object": ObjectWrapper(self._object)}, **self._np_arrays}
            func(self._file_name, **d, allow_pickle=True)
        elif self._backend == "shared_array" or self._backend == "python":
            # save object to file, np arrays to shared memory
            np.savez(self._file_name, object=ObjectWrapper(self._object), allow_pickle=True)
            for name, array in self._np_arrays.items():
                array_to_shared_memory(name, array, self._backend)

            TMP_FILES_IN_SESSION.append(self._file_name)
            SHARED_MEMORIES_IN_SESSION.append(self._file_name)
        else:
            assert False


    def add_np_array(self, array):
        name = self._file_name + "__" + random_name()  # give a name that is a subname of our base name
        self._np_arrays[name] = array
        return name


def object_to_shared_memory(object, base_name=None, backend="shared_array"):
    if base_name is None:
        base_name = random_name()

    data_bundle = DataBundle(object, base_name, backend)

    def _replace_func(o):
        if isinstance(o, np.ndarray):
            # wrap numpy arrays
            name = data_bundle.add_np_array(o)
            return _Wrapper(name)
        return o

    new_object = replace_object_attributes_recursively(object, _replace_func)

    # write this new object
    data_bundle._object = new_object
    data_bundle.save()
    return base_name


def object_from_shared_memory(base_name, backend="shared_array"):
    if not base_name.endswith(".npz"):
        base_name = base_name + ".npz"

    data = np.load(base_name, allow_pickle=True)
    object = np.atleast_1d(data["object"])[0].object  # hack: Object is wrapped in object array

    def _replace_func(o):
        if isinstance(o, _Wrapper):
            # wrap numpy arrays
            if backend in ("file", "compressed_file"):
                return data[o.object]
            elif backend == "shared_array" or backend == "python":
                return array_from_shared_memory(o.object, backend)
            else:
                assert False

        return o

    # find numpy arrays and replace
    object = replace_object_attributes_recursively(object, _replace_func, ignore_types=_Wrapper)
    return object


def from_file(name):
    return object_from_shared_memory(name, "file")


def to_file(object, base_name=None, compress=False):
    if base_name is not None and base_name.endswith(".npz"):
        base_name = base_name.replace(".npz", "")

    backend = "file" if not compress else "compressed_file"
    name = object_to_shared_memory(object, base_name, backend=backend)
    logging.info("Wrote to file: %s" % name)
    return name

