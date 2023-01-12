from shared_memory_wrapper import from_file, to_file
from dataclasses import dataclass


@dataclass
class A:
    a: list
    b: str
    c: float


def test():
    a = A([1, 2, 3], "asdf", 3.1)
    to_file(a, "test.tmp")

    a2 = from_file("test.tmp")

    print(a, a2)
    assert a == a2
