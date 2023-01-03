# Shared Memory Wrapper

This package provides simple functionality for making it easier to put more complex Python objects into shared memory. There are already many good ways to put NumPy arrays into shared memory in Python, but when you have arrays wrapped in other more complex objects, things are not straight-forward. This package handles that by traversing objects recusively, finding out what are NumPy arrays and what is not, and then puts all NumPy arrays into shared memory and pickles the rest. This means that the NumPy-part of your data can be in real shared memory, which is usually what you want (the rest is copied using Pickle):

```python
import numpy as np
from shared_memory_wrapper import object_to_shared_memory, object_from_shared_memory

my_object = [[1, 2, 3], np.arange(10), {1: "hi"}]
identifier = object_to_shared_memory(my_object)

# in the same or in some other python process
shared_object = object_from_shared_memory(identifier)

# Add one to the NumPy array:
# This is reflected wherever you have read your shared object
# but not in the original object
shared_object[1] += 1  

# change the first element (list)
# this is not reflected anywhere, since this is not a 
# NumPy array and is thus not really shared
shared_object[0] = "Test"

# Important if you don't want to have arrays taking up memory, always call after finishing:
from shared_memory_wrapper import free_memory, remove_shared_memory_in_session
free_memory()

# Alternatively, remove only the shared memory created in this process:
remove_shared_memory_in_session()
```

This packages also supports writing objects in a similar way to and from file. The same logic is used: Everything that is NumPy is stored as raw NumPy arrays, which is efficient, and the rest is pickled:

```python
from shared_memory_wrapper import to_file, from_file
to_file(my_object, "filename.npz", compress=False)
o = from_file("filename.npz")
```

Caveats:
 * This is quite efficient as long as your objects keep all big data in NumPy arrays. Everything else than NumPy arrays gets pickled, so large dicts, lists etc. will be slow.
 * After putting an object into shared memory, the original object will be "broken" and should not be used anymore. If you want to continue using the object in the process you are in, simply read it from shared memory again to reconstruct it.


### Installation
Requires Python 3.
``` bash
pip install shared_memory_wrapper
```





