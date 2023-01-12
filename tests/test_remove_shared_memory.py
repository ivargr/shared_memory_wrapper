from shared_memory_wrapper import object_to_shared_memory
import numpy as np
import shared_memory_wrapper
from shared_memory_wrapper.shared_memory import SHARED_MEMORIES_IN_SESSION


def test():
    # free all first
    shared_memory_wrapper.free_memory()

    a = np.array([1, 2, 2])
    s = object_to_shared_memory(a)
    print(s)
    print(SHARED_MEMORIES_IN_SESSION)
    assert len(SHARED_MEMORIES_IN_SESSION) == 2
    shared_memory_wrapper.shared_memory.remove_shared_memory(s)
    print(SHARED_MEMORIES_IN_SESSION)
    assert len(SHARED_MEMORIES_IN_SESSION) == 0


