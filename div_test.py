import logging
import time

logging.basicConfig(level=logging.INFO)
from shared_memory_wrapper import from_file
from shared_memory_wrapper.util import log_memory_usage
import numpy as np


index = from_file("../../counter_index_only_variants_with_revcomp.npz")
log_memory_usage("index read done")

kmers = index.kmers

t = time.perf_counter()
for i, kmers in enumerate(np.array_split(kmers, 50)):
    index.count_kmers(kmers)
    log_memory_usage("After counting %d" % i)


print(time.perf_counter()-t)

print(np.sum(index.get_node_counts()))