from __future__ import division
import math

import numpy as np


def reduce_(in_data, thread_count=4, operator=lambda x, y: x + y):
    '''
    >>> data = np.array(range(10))
    >>> data
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    The default reduction operator is sum:

    >>> reduce_(data)
    45
    >>> data.sum()
    45

    However, we can specify a different operator using the 'operator'
    keyword argument.

    For example, we can easily find the minimum value in a list using:

    >>> reduce_(data, operator=lambda x, y: min(x, y))
    0
    >>> data.min()
    0

    Or, the maximum value:

    >>> reduce_(data, operator=lambda x, y: max(x, y))
    9
    >>> data.max()
    9
    '''
    iter(in_data)
    data = np.array(in_data)
    
    remaining_size = len(data)
    thread_count = min(remaining_size, thread_count)

    passes = int(math.ceil(remaining_size / thread_count))
    stride = thread_count

    for tid in range(thread_count):
        for pass_id in range(passes):
            index = tid + (pass_id + 1) * stride
            if index < remaining_size:
                data[tid] = operator(data[tid], data[index])

    remaining_size = thread_count

    while remaining_size > 1:
        half_size = int(math.ceil(remaining_size / 2))
        for tid in range(thread_count):
            index = tid + half_size
            if index < remaining_size:
                data[tid] = operator(data[tid], data[index])
        remaining_size = half_size
    return data[0]
