import copy
import logging
logging.basicConfig(level=logging.INFO)
import pytest
import numpy as np
from shared_memory_wrapper.shared_memory import np_array_to_shared_memory, np_array_from_shared_memory, remove_all_shared_memory, _get_object_init_arguments
np.random.seed(1)
from shared_memory_wrapper import to_shared_memory, from_shared_memory, free_memory, free_memory_in_session, remove_shared_memory_in_session
from shared_memory_wrapper import object_to_shared_memory, object_from_shared_memory


def test_to_and_from_shared_memory():
    a = np.random.randint(0, 1000, 1000)
    np_array_to_shared_memory(a, "test")

    b = np_array_from_shared_memory("test")
    assert np.all(a == b)

    remove_all_shared_memory()

class A:
    def __init__(self, a, b, c=3):
        self.a = a
        self._b = b  # private variable on purpose
        self.c = c


def test_object_init_arguments():
    a = A(1, 2, 3)
    arguments = _get_object_init_arguments(a)
    assert arguments == ["a", "b", "c"]


def test_object_to_from_shared_memory():
    object = A(np.array([1, 2, 3]), np.array([10.5, 3.0]), np.array([10, 10], dtype=np.uint8))
    true = copy.deepcopy(object)
    name = object_to_shared_memory(object)

    object2 = object_from_shared_memory(name)
    object3 = object_from_shared_memory(name)

    assert object2.c.dtype == np.uint8
    assert np.all(object2.a == true.a)
    object2.a[0] = 0
    assert object3.a[0] == 0
    print("Done")

    remove_shared_memory_in_session()


@pytest.mark.skip("Python not supported")
def test_object_to_from_shared_memory_using_python_backend():
    object = A(np.array([1, 2, 3]), np.array([10.5, 3.0]), np.array([10, 10], dtype=np.uint8))
    to_shared_memory(object, "test3", use_python_backend=True)

    object2 = from_shared_memory(A, "test3", use_python_backend=True)
    object3 = from_shared_memory(A, "test3", use_python_backend=True)

    assert object2.c.dtype == np.uint8
    assert np.all(object2.a == object.a)
    object2.a[0] = 0
    assert object3.a[0] == 0
    print("Done")

def test_to_shared_memory_without_name():
    object = A(np.array([1, 2, 3]), np.array([10.5, 3.0]), np.array([10, 10], dtype=np.uint8))
    name = to_shared_memory(object)
    object2 = from_shared_memory(A, name)
    assert np.all(object2.a == object.a)
    


@pytest.fixture(scope="function", autouse=True)
def cleanup(request):
    yield  # pytest will run tests
    remove_shared_memory_in_session()


