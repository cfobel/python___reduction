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


def reduce_inplace(in_data, thread_count=4, operator='sum', dtype=None):
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
    ''', no_extern_c=True, options=['-I%s/pycuda_include' % module_root()],)

    if dtype is None:
        dtype = in_data.dtype
    assert(dtype in [np.int32, np.float32])

    operator_func_map = {
            np.float32: dict(
                    sum='reduce_float_sum',
                    product='reduce_float_product',
                    min='reduce_float_min',
                    max='reduce_float_max',),
            np.int32: dict(
                    sum='reduce_int_sum',
                    product='reduce_int_product',
                    min='reduce_int_min',
                    max='reduce_int_max',)
    }

    test = mod.get_function(operator_func_map[dtype][operator])

    data = np.array(in_data, dtype=dtype)

    test(np.int32(len(data)), drv.InOut(data), block=(4, 1, 1))

    return data[0]
