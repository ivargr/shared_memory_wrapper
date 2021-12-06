# Shared Memory Wrapper

A very simple wrapper around the Python package [Shared Array](https://pypi.org/project/SharedArray/). Consists of a few helper functions and classer to make it easier to use the Shared Array package on objects with multiple numpy arrays.

### Installation
Requires Python 3.
``` bash
pip install shared_memory_wrapper
```

### How to use?
```python
from shared_memory_wrapper import from_shared_memory, to_shared_memory, free_memory, remove_shared_memory_in_session
```

All objects that follow these rules can be put into shared memory:
* All arguments to the init method must be numpy ndarrays or ints.
* All these variables must be stored in instance variables that match their argument names (alternatively with an underscore first).

Example:
```python
import numpy as np

class SomeClass(a, b, c=None):
    self.a = a
    self._b = b
    self._c  = c

o = Someclass(np.array([1, 2, 3]), np.array([1, 2]), 5)
to_shared_memory(o, "myname")

# in another process
o = from_shared_memoryh(SomeClass, "myname")

# Important if you don't want to have arrays taking up memory, always call after finishing:
free_memory()

# Alternatively, remove only the shared memory created in this process:
remove_shared_memory_in_session()
```

