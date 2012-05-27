from nose.tools import eq_, nottest
import numpy as np

from ..reduction import reduce_


@nottest
def do_test(size, num_threads, reduce_op, op, dtype=None):
    if dtype:
        data = np.array(range(size), dtype=dtype)
    else:
        data = np.array(range(size))

    eq_(reduce_(data, num_threads, reduce_op), op(data))


def test():
    for reduce_op, op in [(lambda x, y: x + y, sum),
            (lambda x, y: x * y, np.prod),
            (lambda x, y: min(x, y), min),
            (lambda x, y: max(x, y), max)]:
        for threads in [4, 8, 16, 32, 64]:
            for size in [4, 10, 32, 127, 263]:
                yield do_test, size, threads, reduce_op, op, np.uint64
