from __future__ import division
import math

from path import path
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule


def module_root():
    '''
    Return absolute path to pyvpr root directory.
    '''
    try:
        script = path(__file__)
    except NameError:
        script = path(sys.argv[0])
    return script.parent.abspath()


def reduce_inplace(in_data, thread_count=128, elements_per_thread=8,
        operator='sum', dtype=None):
    mod = SourceModule(r'''
    #include "reduction.hpp"

    using namespace reduction;

    extern "C" __global__ void reduce_float_sum(int size, float *data);
    extern "C" __global__ void reduce_float_product(int size, float *data);
    extern "C" __global__ void reduce_float_min(int size, float *data);
    extern "C" __global__ void reduce_float_max(int size, float *data);
    extern "C" __global__ void reduce_int_sum(int size, int *data);
    extern "C" __global__ void reduce_int_product(int size, int *data);
    extern "C" __global__ void reduce_int_min(int size, int *data);
    extern "C" __global__ void reduce_int_max(int size, int *data);


    __global__ void reduce_float_sum(int size, float *data) {
        reduce<float, SUM>(size, data);
    }


    __global__ void reduce_float_product(int size, float *data) {
        reduce<float, PRODUCT>(size, data);
    }


    __global__ void reduce_float_min(int size, float *data) {
        reduce<float, MIN>(size, data);
    }


    __global__ void reduce_float_max(int size, float *data) {
        reduce<float, MAX>(size, data);
    }


    __global__ void reduce_int_sum(int size, int *data) {
        reduce<int, SUM>(size, data);
    }


    __global__ void reduce_int_product(int size, int *data) {
        reduce<int, PRODUCT>(size, data);
    }


    __global__ void reduce_int_min(int size, int *data) {
        reduce<int, MIN>(size, data);
    }


    __global__ void reduce_int_max(int size, int *data) {
        reduce<int, MAX>(size, data);
    }


    extern "C" __global__ void global_reduce_int_sum(int size, int *data) {
        global_reduce<int, SUM>(size, data, data);
    }


    extern "C" __global__ void global_reduce_int_product(int size, int *data) {
        global_reduce<int, PRODUCT>(size, data, data);
    }


    extern "C" __global__ void global_reduce_int_min(int size, int *data) {
        global_reduce<int, MIN>(size, data, data);
    }


    extern "C" __global__ void global_reduce_int_max(int size, int *data) {
        global_reduce<int, MAX>(size, data, data);
    }


    extern "C" __global__ void global_reduce_float_sum(int size, float *data) {
        global_reduce<float, SUM>(size, data, data);
    }


    extern "C" __global__ void global_reduce_float_product(int size, float *data) {
        global_reduce<float, PRODUCT>(size, data, data);
    }


    extern "C" __global__ void global_reduce_float_min(int size, float *data) {
        global_reduce<float, MIN>(size, data, data);
    }


    extern "C" __global__ void global_reduce_float_max(int size, float *data) {
        global_reduce<float, MAX>(size, data, data);
    }
    ''', no_extern_c=True, options=['-I%s/pycuda_include' % module_root()],
            keep=True)

    if dtype is None:
        dtype = in_data.dtype
    assert(dtype in [np.int32, np.float32])

    operator_func_map = {
            np.float32: dict(
                    global_sum='global_reduce_float_sum',
                    global_product='global_reduce_float_product',
                    global_min='global_reduce_float_min',
                    global_max='global_reduce_float_max',
                    sum='reduce_float_sum',
                    product='reduce_float_product',
                    min='reduce_float_min',
                    max='reduce_float_max',),
            np.int32: dict(
                    global_sum='global_reduce_int_sum',
                    global_product='global_reduce_int_product',
                    global_min='global_reduce_int_min',
                    global_max='global_reduce_int_max',
                    sum='reduce_int_sum',
                    product='reduce_int_product',
                    min='reduce_int_min',
                    max='reduce_int_max',)
    }

    try:
        test = mod.get_function(operator_func_map[dtype][operator])
    except drv.LogicError:
        print dtype, operator, operator_func_map[dtype][operator]
        raise

    data = np.array(in_data, dtype=dtype)

    if operator.startswith('global_'):
        shared = np.float32(0).itemsize * thread_count * elements_per_thread
        block_count = int(np.ceil(data.size / thread_count / elements_per_thread))
    else:
        shared = 0
        block_count = 1

    block = (thread_count, 1, 1)
    grid = (block_count, 1, 1)

    test(np.int32(len(data)), drv.InOut(data), block=block, grid=grid,
            shared=shared)

    return data[:block_count]
