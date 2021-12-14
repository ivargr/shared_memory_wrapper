import logging
logging.basicConfig(level=logging.INFO)
import time
import numpy as np
#from shared_memory_wrapper.python_shared_memory import
from shared_memory_wrapper.shared_memory import from_shared_memory, to_shared_memory, free_memory_in_session
from graph_kmer_index import KmerIndex
from kmer_mapper.parser import OneLineFastaParser, KmerHash


class Data:
    def __init__(self, b, c):
        self.b = b
        self.c = c


def get_benchmark_data():
    a = np.random.randint(0, 2**64, 23000000, dtype=np.uint64)
    b = np.random.randint(0, 100000, 200000000, dtype=np.uint64)
    c = np.random.randint(0, 10, 200000000, dtype=np.uint32)

    return a, Data(b, c)


def function1(a, data, modulo=200000000):
    t = time.perf_counter()
    a = a % modulo
    pos = data.b[a]
    pos2 = data.b[a] + data.c[a]
    print("Running function %.6f sec" % (time.perf_counter()-t))
    return pos, pos2

def function2(kmers, index):
    t = time.perf_counter()
    kmer_hashes = kmers % 80000000  # index._modulo
    from_indexes = index._hashes_to_index[kmer_hashes]
    to_indexes = from_indexes + index._n_kmers[kmer_hashes]
    logging.info("Took %.5f sec to get indexes" % (time.perf_counter()-t))
    return from_indexes, to_indexes


def get_kmers():
    fasta_parser = OneLineFastaParser("hg002_simulated_reads_15x.fa", 500000 * 150 // 3)
    reads = fasta_parser.parse(as_shared_memory_object=False)
    for sequence_chunk in reads:
        t = time.perf_counter()
        hashes, reverse_complement_hashes, mask = KmerHash(k=31).get_kmer_hashes(sequence_chunk)
        return hashes


def benchmark1():
    i = KmerIndex.from_file("kmer_index_only_variants.npz")
    a, data = get_benchmark_data()
    kmers = get_kmers()
    print(kmers, kmers.dtype)
    print(a, a.dtype)

    to_shared_memory(i, "index")
    i2 = from_shared_memory(KmerIndex, "index")


    to_shared_memory(i, "index2", use_python_backend=True)
    i3 = from_shared_memory(KmerIndex, "index2", use_python_backend=True)
    function2(a, i)
    function2(a, i2)
    function2(a, i2)
    function2(a, i3)

    function1(a, data)

    to_shared_memory(data, "test")
    data2 = from_shared_memory(Data, "test")
    function1(a, data2)

    to_shared_memory(data, "test2", use_python_backend=True)
    data3 = from_shared_memory(Data, "test2", use_python_backend=True)
    function1(a, data3)
    function1(a, data2, modulo=200000000)
    function1(a, data3, modulo=200000000)

benchmark1()


free_memory_in_session()