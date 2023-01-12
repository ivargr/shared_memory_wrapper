import os
import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

import shared_memory_wrapper.shared_memory

import pytest
from shared_memory_wrapper.shared_memory_v2 import object_from_shared_memory, object_to_shared_memory, to_file, from_file
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
    object_to_shared_memory(a, "testfile", "file")
    a3 = object_from_shared_memory("testfile", "file")
    print(a3, a)
    assert a == a3


def test_to_from_file_nparray():
    a = A(np.random.randint(0, 1000, 10000), "test")
    true = copy.copy(a)
    to_file(a, "test123", "file")
    a3 = object_from_shared_memory("test123", "file")
    assert np.all(true.a == a3.a)


def test_to_from_shared_memory_nparray():
    a = np.random.randint(0, 1000, 10000)
    true = a.copy()
    a2 = object_to_shared_memory(a, None, "shared_array")
    a3 = object_from_shared_memory(a2, "shared_array")
    assert np.all(true == a3)




@pytest.mark.parametrize("data",
                         [
                         [None, 0, 1.5, "Hi"],
                         A([1, 2, 3], 1.5),
                         A([[[1], [1.0]], 1], {1: "hei", 2: [1, 2, 3]}),
                         A(A(1, 1), "test")
])
@pytest.mark.parametrize("backend", ["shared_array", "file", "compressed_file"])
def test_div_objects_div_backends(data, backend):
    true = copy.deepcopy(data)
    name = object_to_shared_memory(data, None, backend)
    data2 = object_from_shared_memory(name, backend)

    assert true == data2

    if backend == "file" or backend == "compressed_file":
        os.remove(name + ".npz")


def test_index_bundle():
    from kage.indexing.index_bundle import IndexBundle
    bundle = IndexBundle({"a": np.array([1, 2, 3])})
    to_file(bundle, "testfile")
    bundle2 = from_file("testfile")
    assert np.all(bundle2.a == [1, 2, 3])



class B:
    def __init__(self, b):
        self.b = b

    def __getattr__(self, item):
        raise AttributeError()


def test_pickle():
    b = B(2)
    np.savez("test.npz", object=b, allow_pickle=True)
    b2 = np.load("test.npz", allow_pickle=True)
    print(b2["object"])



@pytest.fixture(scope="function", autouse=True)
def cleanup(request):
    yield  # pytest will run tests
    remove_shared_memory_in_session()




