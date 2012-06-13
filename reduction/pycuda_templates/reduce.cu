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
