#include "jetson-utils/cudaUtility.h"
#include "iostream"

__global__ void add_single_thread(int n, float *x, float *y) {
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

cudaError_t addWithCuda_single_thread(int n, float *x, float *y) {
    // 只用一個 thread 去跑
    add_single_thread<<<1, 1>>>(n, x, y);
    return CUDA(cudaGetLastError());
}

__global__ void add_multi_thread(int n, float *x, float *y) {
    // threadIdx.x contains the index of the current thread within its block
    // blockDim.x contains the number of threads in the block.
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

cudaError_t addWithCuda_multi_thread(int n, float *x, float *y) {
    // <<< 1, 256 >>> 語法稱為 execution configuration
    // 256 代表 block 的維度 -> 一個 thread block 中有 256 個 thread
    // 1   代表 grid 的維度  -> 一個 grid 中有 1 個 block
    // 告訴 CUDA runtime 用 256 個平行的 thread launch kernel，通常會是 32 的倍數
    add_multi_thread<<<1, 256>>>(n, x, y);
    return CUDA(cudaGetLastError());
}

__global__ void add_multi_block(int n, float *x, float *y) {
    // gridDim.x, which contains the number of blocks in the grid,
    // and blockIdx.x, which contains the index of the current block within the grid.
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // total number of threads in the grid
    int stride = blockDim.x * gridDim.x;

    // a grid-stride loop.
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

cudaError_t addWithCuda_multi_block(int n, float *x, float *y) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    add_multi_block<<<numBlocks, blockSize>>>(n, x, y);
    return CUDA(cudaGetLastError());
}