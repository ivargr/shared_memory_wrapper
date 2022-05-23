
def interval_chunks(start, end, n_chunks):
    assert end > start
    boundaries = list(range(start, end, n_chunks))
    boundaries.append(end)
    return [(start, end) for start, end in zip(boundaries[0:-1], boundaries[1:])]




