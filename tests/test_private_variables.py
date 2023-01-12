import copy
import logging
logging.basicConfig(level=logging.DEBUG)
from shared_memory_wrapper.shared_memory_v2 import object_to_shared_memory, object_from_shared_memory
from shared_memory_wrapper import remove_shared_memory_in_session
import numpy as np
import pytest

class A:
    def __init__(self, a, b):
        self._a = a
        self.__b = b

    def get_b(self):
        return self.__b

    def __str__(self):
        return self._a + self.__b

def test():
    a = A("1", "2")
    m = object_to_shared_memory(a)
    a2 = object_from_shared_memory(m)
    assert a.get_b() == a2.get_b()


def test2():
    import npstructures as nps

    true = nps.raggedarray.base.RaggedBase(np.array([1, 2, 3]), nps.RaggedShape(np.array([3])))
    r = nps.raggedarray.base.RaggedBase(np.array([1, 2, 3]), nps.RaggedShape(np.array([3])))

    r2 = object_to_shared_memory(r)
    r3 = object_from_shared_memory(r2)

    print(true._RaggedBase__data)
    print(r3._RaggedBase__data)

    assert np.all(true._RaggedBase__data == r3._RaggedBase__data)


def test3():
    import npstructures as nps
    r = nps.RaggedArray([1, 2, 3], [3])
    true = copy.deepcopy(r)
    r2 = object_to_shared_memory(r)
    r3 = object_from_shared_memory(r2)

    assert np.all(true == r3)


@pytest.fixture(scope="function", autouse=True)
def cleanup(request):
    yield  # pytest will run tests
    remove_shared_memory_in_session()

