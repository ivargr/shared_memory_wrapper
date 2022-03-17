# Shared Memory Wrapper

A very simple wrapper around the Python package [Shared Array](https://pypi.org/project/SharedArray/). Consists of a few helper functions and classer to make it easier to use the Shared Array package on objects with multiple numpy arrays.

The aim of this package is to enable sharing many types of Python objects in memory. When objects contain Numpy arrays, these will be very efficiently shared using the Shared Array package. Other objects like strings, ints etc are simply pickled (not really shared in memory).


### Installation
Requires Python 3.
``` bash
pip install shared_memory_wrapper
```

### How to use?
```python
from shared_memory_wrapper import object_to_shared_memory, object_from_shared_memory
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
object_to_shared_memory(o, "myname")

# in another process
object_from_shared_memory(SomeClass, "myname")

# Important if you don't want to have arrays taking up memory, always call after finishing:
free_memory()

# Alternatively, remove only the shared memory created in this process:
remove_shared_memory_in_session()
```

This package also supports writing many types of objects to and from file. Example:

```python
from shared_memory_wrapper import to_file, from_file

class SomeObject:
    def __init__(self, a, b):
        self._a = a
        self._b = b

class MyObject:
    def __init__(self, some_string, some_list, some_object):
        self._some_string = some_string
        self._some_list = some_list
        self._some_object = some_ombject


o = MyObject("test", [1, 2, 3], SomeObject(np.ndarray([1, 2, 3]), 5.0))

to_file("o", "testfile")
o2 = from_file("testfile")
```



