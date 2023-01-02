import logging
import types

import numpy as np


def replace_object_attributes_recursively(object, func, ignore_types=None):
    """
    Runs replace_func on an object and its "children" recursively.
    Can be used to e.g. search for specific data and replace it.
    """

    if ignore_types is not None and isinstance(object, ignore_types):
        # dont unpack types in ignore_types
        return func(object)
    elif object is None:
        return object
    elif isinstance(object, (str, int, float, np.integer)):
        # don't do anything with primitive types
        return func(object)
    elif isinstance(object, np.ndarray):
        # don't do anything with numpy arrays
        res = func(object)
        return res
    elif isinstance(object, tuple):
        # convert to list, replace, and convert back
        return tuple(replace_object_attributes_recursively(list(object), func, ignore_types))
    elif isinstance(object, set):
        # just pickle set
        return object
    elif isinstance(object, dict):
        # run on each value
        return {key: replace_object_attributes_recursively(value, func, ignore_types) for key, value in object.items()}
    elif isinstance(object, list):
        # run on each element
        return [replace_object_attributes_recursively(e, func, ignore_types) for e in object]
    elif isinstance(object, np.dtype):
        return object
    elif callable(object):   #isinstance(object, types.FunctionType):
        #print("Object is callable: ", object)
        return object
    else:
        # assume regular object, iterate all attributes
        try:
            items = object.__dict__.items()
        except AttributeError:
            print(object)
            raise

        for attr, value in items:
            setattr(object, attr, replace_object_attributes_recursively(value, func, ignore_types))

        return func(object)

