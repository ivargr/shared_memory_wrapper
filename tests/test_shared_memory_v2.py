import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
import pytest
from shared_memory_wrapper.shared_memory_v2 import object_from_shared_memory, object_to_shared_memory
import numpy as np
import copy
from shared_memory_wrapper import remove_shared_memory_in_session


class A:
    def __init__(self, a, b):
        self.a = a
        self._b = b
        self.__c = "Hei"

    def __eq__(self, other):
        return self.a == other.a and self._b == other._b and self._A__c == other._A__c


def test_to_from_shared_memory_list():
    a = [1, 2, 3]
    a2 = object_to_shared_memory(a, None, "file")
    a3 = object_from_shared_memory(a2, "file")
    print(a3, a)
    assert a == a3


def test_to_from_file_nparray():
    a = np.random.randint(0, 1000, 10000)
    a2 = object_to_shared_memory(a, "test", "file")
    a3 = object_from_shared_memory(a2, "file")
    assert np.all(a == a3)


def test_to_from_shared_memory_nparray():
    a = np.random.randint(0, 1000, 10000)
    true = a.copy()
    a2 = object_to_shared_memory(a, None, "shared_array")
    a3 = object_from_shared_memory(a2, "shared_array")
    assert np.all(true == a3)




@pytest.mark.parametrize("data",
                         [None, 1, 1.5, "Hi",
                         A([1, 2, 3], 1.5),
                         A([[[1], [1.0]], 1], {1: "hei", 2: [1, 2, 3]}),
                         A(A(1, 1), "test")
])
@pytest.mark.parametrize("backend", ["shared_array", "file", "compressed_file"])
def test(data, backend):
    true = copy.deepcopy(data)
    name = object_to_shared_memory(data, None, backend)
    data2 = object_from_shared_memory(name, backend)

    assert true == data2
    remove_shared_memory_in_session()


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    print("CLEANUP!")
    print("Finished")
    remove_shared_memory_in_session()



