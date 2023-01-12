import logging
logging.basicConfig(level=logging.INFO)
import pytest
from shared_memory_wrapper.util import interval_chunks
import numpy as np
from dataclasses import dataclass
import multiprocessing
from shared_memory_wrapper.util import Mapper, Reducer, SumReducer
from shared_memory_wrapper.util import interval_chunks
from shared_memory_wrapper.util import parallel_map_reduce
from shared_memory_wrapper import remove_shared_memory_in_session


def test_interval_chunks():
    assert interval_chunks(0, 3, 3) == [(0, 1), (1, 2), (2, 3)]
    assert interval_chunks(0, 10, 3) == [(0, 3), (3, 6), (6, 9), (9, 10)]
    assert interval_chunks(1, 2, 3) == [(1, 2)]



def process_single_number(number):
    return number*number


def process_numbers(numbers):
    sum = 0
    for number in numbers:
        sum += process_single_number(number)

    return sum

def some_function(a, b, interval):
    return np.sum(a[interval[0]:interval[1]] * b[interval[0]:interval[1]])


def test_map_reduce():
    a = np.arange(10)
    b = np.arange(10, 20)

    truth = np.sum(a * b)

    result = parallel_map_reduce(some_function, (a, b), mapper=interval_chunks(0, 10, 3), reducer=SumReducer())

    assert truth == result


def add(matrix, matrix2, i):
    matrix += matrix2
    matrix += i

def add_nonparallell(matrix, matrix2, interval):
    for i in range(*interval):
        add(matrix, matrix2, i)


def test_matrix_addition():
    dimension = 5
    matrix = np.zeros((dimension, dimension))
    matrix2 = np.arange(dimension*dimension).reshape((dimension, dimension))

    print(matrix)
    print(matrix2)
    n_threads = 8
    final_matrix, final_matrix2 = parallel_map_reduce(add_nonparallell, (matrix, matrix2),
                                                      mapper=interval_chunks(0, 100, n_threads), n_threads=n_threads)

    print(final_matrix)
    add_nonparallell(matrix, matrix2, (0, 100))
    #assert np.all(final_matrix == matrix)


def adder2(matrix, interval):
    for i in range(*interval):
        return matrix + i


def test_map_reduce_matrix_adder():
    inital = np.zeros((10, 10))



@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    print("cleanup!")
    print("finished")
    remove_shared_memory_in_session()

