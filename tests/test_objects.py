import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from shared_memory_wrapper.shared_memory import object_to_shared_memory, object_from_shared_memory
from shared_memory_wrapper import free_memory_in_session


class B:
    def __init__(self, array):
        self._array = array

    def __eq__(self, other):
        return np.all(self._array == other._array)


class A:
    def __init__(self, number, object, array):
        self._number = number
        self._object = object
        self._array = array

    def __eq__(self, other):
        return self._number == other._number and \
            self._object == other._object and \
            np.all(self._array == other._array)


class C:
    def __init__(self, string, b):
        self._string = string
        self.b = b

    def __eq__(self, other):
        return self._string == other._string and self.b == other.b


def _get_dummy_object():
    return A(100, A(100, C("hello", 3.0), np.array([1])), np.array([1, 2, 3], dtype=float))


def test():
    a = A(5, B(np.array([1, 2, 3])), np.array([4, 3, 1]))
    name = object_to_shared_memory(a)
    a2 = object_from_shared_memory(name)

    assert np.all(a2._number == a._number)
    assert np.all(a2._object._array == a._object._array)
    assert np.all(a2._array == a._array)


def test2():
    a = A(100, A(100, C("hello", 3.0), np.array([1])), np.array([1, 2, 3], dtype=float))
    name = object_to_shared_memory(a)

    a2 = object_from_shared_memory(name)

    assert a2._object._number == 100
    assert a2._object._object.b == 3.0
    a2._object._array[0] = 10
    assert a2._object._array[0] == 10

def test_counter():
    from npstructures import Counter, HashTable
    from npstructures.raggedarray import RaggedArray, RaggedShape


    counter = Counter([1,2, 3])
    name = object_to_shared_memory(counter)
    counter2 = object_from_shared_memory(name)

    counter.count([1, 2, 3])
    counter2.count([1,  2, 3])

    assert np.all(counter[1, 2, 3] == counter2[1, 2, 3])


def test_various_backends():

    for backend in ["shared_array", "python", "file"]:
        print(backend)
        a = _get_dummy_object()
        name = object_to_shared_memory(a, backend="file")
        a2 = object_from_shared_memory(name, backend="file")
        assert a2 == a


test()
test2()
test_counter()
test_various_backends()

free_memory_in_session()

