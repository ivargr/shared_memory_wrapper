import logging
logging.basicConfig(level=logging.INFO)
import pytest
import os
import numpy as np
#from shared_memory_wrapper.shared_memory import object_to_shared_memory, object_from_shared_memory
from shared_memory_wrapper.shared_memory_v2 import object_to_shared_memory, object_from_shared_memory
from shared_memory_wrapper import free_memory_in_session
from shared_memory_wrapper import remove_shared_memory_in_session
from shared_memory_wrapper.shared_memory_v2 import to_file, from_file
from shared_memory_wrapper.shared_memory import TMP_FILES_IN_SESSION
import copy


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
    true = copy.deepcopy(a)
    name = object_to_shared_memory(a)
    a2 = object_from_shared_memory(name)

    assert np.all(a2._number == true._number)
    assert np.all(a2._object._array == true._object._array)
    assert np.all(a2._array == true._array)


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
    true = copy.deepcopy(counter)
    name = object_to_shared_memory(counter)
    counter2 = object_from_shared_memory(name)

    true.count([1, 2, 3])
    counter2.count([1,  2, 3])

    assert np.all(true[1, 2, 3] == counter2[1, 2, 3])


def test_various_backends():
    for backend in ["shared_array", "file"]:
        print(backend)
        a = _get_dummy_object()
        true = copy.deepcopy(a)
        name = object_to_shared_memory(a, backend=backend)
        a2 = object_from_shared_memory(name, backend=backend)
        assert a2 == true

        if backend == "file":
            os.remove(name + ".npz")


def test_to_from_file():
    object = _get_dummy_object()
    to_file(object, "testobject")
    object2 = from_file("testobject")

    assert _get_dummy_object() == object2



def test_kmer_index_counter():
    from graph_kmer_index import CounterKmerIndex
    from npstructures import Counter

    kmers = np.array([1, 2, 3, 4], dtype=np.int64)
    nodes = np.array([10, 20, 30, 40], dtype=np.uint32)
    unique_kmers = np.unique(kmers)
    counter = Counter(unique_kmers, np.zeros_like(unique_kmers))
    #counter.count([1])
    #counter.count([2, 2, 3])
    index = CounterKmerIndex(kmers, nodes, counter)
    print(index.counter._values)
    counter.count([1])
    print(index.counter._values)
    #index.counter.count([1])
    to_file(index, "testindex")
    index2 = object_to_shared_memory(from_file("testindex"))


    index3 = object_from_shared_memory(index2)
    index3.counter.count([1, 2, 2, 3])

    assert np.all(index3.counter[1, 2, 3, 4] == np.array([2, 2, 1, 0]))
    print(index3.counter)

    index4 = object_from_shared_memory(index2)
    print(index4.counter)

    index4.counter.count([1])
    index5 = object_from_shared_memory(index2)
    print(index4.counter, index5.counter, index3.counter)



def test_list_object():
    object = [1, np.array([1, 2, 3]), 3]
    name = object_to_shared_memory(object)
    object2 = object_from_shared_memory(name)

    assert object2[0] == 1
    assert np.all(object2[1] == [1, 2, 3])
    assert object2[2] == 3


def test_single_base_types():
    name = object_to_shared_memory("test")
    assert object_from_shared_memory(name) == "test"
    assert object_from_shared_memory(object_to_shared_memory(5.1)) == 5.1


def test_list():
    name = object_to_shared_memory([1, 2, 3])
    assert object_from_shared_memory(name) == [1, 2, 3]

    name = object_to_shared_memory([1, B("hei"), 3])
    assert object_from_shared_memory(name)[1]._array == "hei"


class D:
    def __init__(self, l):
        self.l = l


def test_object_with_list():
    d = D([np.uint32(1)])
    d2 = object_from_shared_memory(object_to_shared_memory(d))
    assert d.l == d2.l


def test_tuple():
    a = (5, 3.0, "a")
    a2 = object_from_shared_memory(object_to_shared_memory(a))

    assert a == a2


def test_dict():
    d = {"test": 1, "test2": "hi"}
    true = d.copy()
    name = object_to_shared_memory(d)
    d2 = object_from_shared_memory(name)
    assert true == d2


def __test_multi_hashtable():
    from npstructures.multi_value_hashtable import MultiValueHashTable
    h = MultiValueHashTable.from_keys_and_values(np.array([1, 2, 3]), {"key1": np.array([1, 2, 1]), "key2": np.array([5, 6, 7])})
    h2 = object_from_shared_memory(object_to_shared_memory(h))
    print(h, h2)
    print(h[1], h2[1])
    assert h[1]["key1"] == h2[1]["key1"]
    assert h[3] == h2[3]


def test_set():
    a = set([1, 5, 3, 4])
    assert object_from_shared_memory(object_to_shared_memory(a)) == a



def test_obgraph():
    from obgraph import Graph
    graph = Graph.from_dicts(
        {1: "ACT", 2: "T", 3: "G", 4: "TGTTTAAA"},
        {1: [2, 3], 2: [4], 3: [4]},
        [1, 2, 4]
    )
    true = copy.deepcopy(graph)

    g = object_to_shared_memory(graph)
    g2 = object_from_shared_memory(g)

    assert g2.chromosome_start_nodes == true.chromosome_start_nodes
    assert g2 == true


def test_compressed_file():
    from obgraph import Graph
    a = A(100, A(100, C("hello", 3.0), np.array([1])), np.array([1, 2, 3], dtype=float))
    true = copy.deepcopy(a)

    to_file(a, "test.tmp", compress=True)
    a2 = from_file("test.tmp")

    assert true == a2



@pytest.fixture(scope="function", autouse=True)
def cleanup(request):
    yield  # pytest will run tests
    remove_shared_memory_in_session()

