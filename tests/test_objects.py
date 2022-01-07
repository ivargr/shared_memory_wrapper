import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from shared_memory_wrapper.shared_memory import object_to_shared_memory, object_from_shared_memory


class B:
    def __init__(self, array):
        self._array = array

class A:
    def __init__(self, number, object, array):
        self._number = number
        self._object = object
        self._array = array



class C:
    def __init__(self, string, b):
        self._string = string
        self.b = b


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
    assert a2._object._array[0] == 1

test()
test2()




