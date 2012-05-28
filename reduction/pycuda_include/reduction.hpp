namespace reduction {
    #include <stdio.h>
    #include <stdint.h>
    #include <math.h>

    enum reduce_op {
        SUM,
        PRODUCT,
        MIN,
        MAX,
    };


    template <class T>
    __device__ void dump_data(int size, T *data) {
        syncthreads();
        if(threadIdx.x == 0) {
            printf("[");
            for(int i = 0; i < size; i++) {
                printf("%g, ", data[i]);
            }
            printf("]\n");
        }
    }


    template <class T, reduce_op op>
    __device__ T reduce(int size, T *data) {
        int remaining_size = size;
        int thread_count = min(remaining_size, blockDim.x);
        uint32_t tid = threadIdx.x;

        int passes = ceil(float(remaining_size) / thread_count);
        int stride = thread_count;
        int index;

        for(int pass_id = 0; pass_id < passes; pass_id++) {
            index = tid + (pass_id + 1) * stride;
            if(index < remaining_size) {
                if(op == SUM) {
                    data[tid] = data[tid] + data[index];
                } else if(op == PRODUCT) {
                    data[tid] = data[tid] * data[index];
                } else if(op == MIN) {
                    data[tid] = min(data[tid], data[index]);
                } else if(op == MAX) {
                    data[tid] = max(data[tid], data[index]);
                }
            }
        }

        remaining_size = thread_count;

        while(remaining_size > 1) {
            int half_size = ceil(float(remaining_size) / 2);
            index = tid + half_size;
            if(index < remaining_size) {
                if(op == SUM) {
                    data[tid] = data[tid] + data[index];
                } else if(op == PRODUCT) {
                    data[tid] = data[tid] * data[index];
                } else if(op == MIN) {
                    data[tid] = min(data[tid], data[index]);
                } else if(op == MAX) {
                    data[tid] = max(data[tid], data[index]);
                }
            }
            remaining_size = half_size;
        }
        return data[0];
    }

}
