from __future__ import division
import math


def reduce(data, thread_count=4):
    data_copy = data[:]
    iter(data)

    total_size = len(data)
    remaining_size = total_size

    print data

    passes = int(math.ceil(remaining_size / thread_count))
    stride = thread_count
    print 'size: %d, stride: %d, passes: %d' % (remaining_size, stride, passes)

    for tid in range(thread_count):
        for pass_id in range(passes):
            index = tid + (pass_id + 1) * stride
            if index < remaining_size:
                print '[pass %d] add data[%d] + data[%d] = %d' % (pass_id,
                        tid, index, data[tid] + data[index])
                data[tid] += data[index]
            else:
                print '[pass %d] index %d >= remaining_size' % (
                        pass_id, index)
    print data

    remaining_size = thread_count

    while remaining_size > 1:
        half_size = int(math.ceil(remaining_size / 2))
        for tid in range(thread_count):
            index = tid + half_size
            if index < remaining_size:
                print '[pass %d] add data[%d] + data[%d] = %d' % (pass_id,
                        tid, index, data[tid] + data[index])
                data[tid] += data[index]
            else:
                print '[pass %d] index %d >= remaining_size' % (
                        pass_id, index)
        print data
        remaining_size = half_size

    print data[0]
    print sum(data_copy)
