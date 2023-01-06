#include <math.h>

#include <chrono>
#include <iostream>

#include "jetson-utils/cudaMappedMemory.h"
#include "jetson-utils/cudaUtility.h"

// function to add the elements of two arrays
void add(int n, float *x, float *y) {
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}
void addWithCuda_single_thread(int n, float *x, float *y);
void addWithCuda_multi_thread(int n, float *x, float *y);
void addWithCuda_multi_block(int n, float *x, float *y);

int main(void) {
    int N = 1 << 20;  // 1M elements

    // ========================================================================
    // CPU
    // ========================================================================
    float *x = new float[N];
    float *y = new float[N];

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // calculate time
    auto start = std::chrono::high_resolution_clock::now();

    // Run kernel on 1M elements on the CPU
    add(N, x, y);

    // calculate time
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Elapsed time CPU: " << elapsed.count() << " s" << std::endl;

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    delete[] x;
    delete[] y;

    // ========================================================================
    // CUDA - Single Thread
    // ========================================================================
    float *a, *b;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&a, N * sizeof(float));
    cudaMallocManaged(&b, N * sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    start = std::chrono::high_resolution_clock::now();

    // Run kernel on 1M elements on the GPU
    addWithCuda_single_thread(N, a, b);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // calculate time
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    std::cout << "Elapsed time CUDA - Single: " << elapsed.count() << " s" << std::endl;

    // Check for errors (all values should be 3.0f)
    maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(b[i] - 3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(a);
    cudaFree(b);

    // ========================================================================
    // CUDA - Multi Thread
    // ========================================================================

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&a, N * sizeof(float));
    cudaMallocManaged(&b, N * sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    start = std::chrono::high_resolution_clock::now();

    // Run kernel on 1M elements on the GPU
    addWithCuda_multi_thread(N, a, b);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // calculate time
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    std::cout << "Elapsed time CUDA - Multi Threads: " << elapsed.count() << " s" << std::endl;

    // Check for errors (all values should be 3.0f)
    maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(b[i] - 3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(a);
    cudaFree(b);

    // ========================================================================
    // CUDA - Multi Blocks
    // ========================================================================

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&a, N * sizeof(float));
    cudaMallocManaged(&b, N * sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    start = std::chrono::high_resolution_clock::now();

    // Run kernel on 1M elements on the GPU
    addWithCuda_multi_block(N, a, b);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // calculate time
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    std::cout << "Elapsed time CUDA - Multi Blocks: " << elapsed.count() << " s" << std::endl;

    // Check for errors (all values should be 3.0f)
    maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(b[i] - 3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(a);
    cudaFree(b);

    return 0;
}