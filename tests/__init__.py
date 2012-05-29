from __future__ import division

from nose.tools import eq_, ok_, nottest
import numpy as np

from ..timeit import timeit
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

    eq_(reduce_inplace(data, thread_count=num_threads, operator=reduce_op,
            dtype=dtype), op(data))


@nottest
def do_cuda_global_test(size, num_threads, reduce_op, op, dtype):
    data = np.array(range(size), dtype=dtype)

    cuda_result, cuda_time = timeit(reduce_inplace, (data, ), dict(
            thread_count=num_threads, operator=reduce_op, dtype=dtype))
    cpu_result, cpu_time = timeit(op, (data, ))

    print 'size:%d, num_threads:%d, reduce_op: %s, dtype: %s, cpu_time: %.2g,'\
            'cuda_time: %.2g' % (size, num_threads, reduce_op, dtype, cpu_time,
                    cuda_time)
    ok_(np.allclose([cuda_result], [cpu_result]))


def test_cpu():
    for reduce_op, op in [(lambda x, y: x + y, sum),
            (lambda x, y: x * y, np.prod),
            (lambda x, y: min(x, y), min),
            (lambda x, y: max(x, y), max)]:
        for threads in [4, 8, 16, 32, 64]:
            for size in [4, 10, 32, 127, 263]:
                yield do_test, size, threads, reduce_op, op, np.uint64


if CUDA_ENABLED:
    def test_cuda_local():
        for reduce_op, op in [('sum', sum), ('product', np.prod), ('min', min),
                ('max', max),]:
            for size in [4, 10, 32, 127, 263]:
                if size > 32 and reduce_op == 'product':
                    break
                for threads in [4, 8, 16, 32, 64]:
                    for dtype in [np.int32, np.float32]:
                        yield do_cuda_test, size, threads, reduce_op, op, dtype


    def test_cuda_global():
        for reduce_op, op in [('global_sum', sum), ('global_product', np.prod),
                ('global_min', min), ('global_max', max),]:
            for size_exponent in range(2, 17):
                size = 2 ** size_exponent
                if size > 32 and reduce_op == 'global_product':
                    break
                for threads in [16, 32, 64, 128, 256, 512]:
                    for dtype in [np.int32, np.float32]:
                        yield do_cuda_global_test, size, threads, reduce_op,\
                                op, dtype
