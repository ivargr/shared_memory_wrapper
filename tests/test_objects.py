import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from shared_memory_wrapper.shared_memory import object_to_shared_memory, object_from_shared_memory
from shared_memory_wrapper import free_memory_in_session
from shared_memory_wrapper import to_file, from_file


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



def test_to_from_file():
    object = _get_dummy_object()
    to_file(object, "testobject")
    object2 = from_file("testobject")

    assert object == object2



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
    index = to_file(index)
    index2 = object_to_shared_memory(from_file(index))


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


def test_dict():
    d = {"test": 1, "test2": "hi"}
    name = object_to_shared_memory(d)
    d2 = object_from_shared_memory(name)

    print(d2)
    assert d == d2


def test_multi_hashtable():
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

#test_kmer_index_counter()
test()
test2()
test_counter()
test_various_backends()
test_to_from_file()
test_list_object()
test_single_base_types()
test_list()
test_dict()
test_multi_hashtable()
test_set()


free_memory_in_session()


