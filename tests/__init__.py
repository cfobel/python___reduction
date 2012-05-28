from nose.tools import eq_, nottest
import numpy as np

from ..reduction import reduce_
try:
    from ..reduction.cuda import reduce_inplace
    CUDA_ENABLED = True
except ImportError:
    CUDA_ENABLED = False
    

@nottest
def do_test(size, num_threads, reduce_op, op, dtype=None):
    if dtype:
        data = np.array(range(size), dtype=dtype)
    else:
        data = np.array(range(size))

    eq_(reduce_(data, num_threads, reduce_op), op(data))


@nottest
def do_cuda_test(size, num_threads, reduce_op, op, dtype):
    data = np.array(range(size), dtype=dtype)

    eq_(reduce_inplace(data, num_threads, reduce_op, dtype), op(data))


def test_cpu():
    for reduce_op, op in [(lambda x, y: x + y, sum),
            (lambda x, y: x * y, np.prod),
            (lambda x, y: min(x, y), min),
            (lambda x, y: max(x, y), max)]:
        for threads in [4, 8, 16, 32, 64]:
            for size in [4, 10, 32, 127, 263]:
                yield do_test, size, threads, reduce_op, op, np.uint64


if CUDA_ENABLED:
    def test_cuda():
        for reduce_op, op in [('sum', sum), ('product', np.prod), ('min', min),
                ('max', max),]:
            for size in [4, 10, 32, 127, 263]:
                if size > 32 and reduce_op == 'product':
                    break
                for threads in [4, 8, 16, 32, 64]:
                    for dtype in [np.int32, np.float32]:
                        yield do_cuda_test, size, threads, reduce_op, op, dtype
