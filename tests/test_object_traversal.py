import numpy as np
from shared_memory_wrapper.object_traversal import replace_object_attributes_recursively


class TestObject:
    def __init__(self, a, b):
        self._a = a
        self.b = b

    def __eq__(self, other):
        return self._a == other._a and self.b == other.b


def test_replace_object_attributes_recursively():
    t = TestObject(1, "test")
    t2 = replace_object_attributes_recursively(t, lambda t: t)
    assert t == t2


def test_replace_object_attributes_recursively_list():
    t = [1, 2, 3, "hei"]
    t2 = replace_object_attributes_recursively(t, lambda t: t)
    print(t2)
    assert t == t2


def test_replace_object_attributes_recursively_nparray():
    t = TestObject("test", np.array([1, 2, 3, 4]))
    t2 = replace_object_attributes_recursively(t, lambda t: t)
    print(t2)
    assert np.all(t.b == t2.b)


def test_replace_object_attributes_recursively_tuple():
    t = TestObject("test", (1, 2))
    t2 = replace_object_attributes_recursively(t, lambda t: t)
    print(t2)
    assert t.b == t2.b