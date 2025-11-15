// matmul_optA.cu
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdlib>
#include <stdio.h>

#define TILE 32

__global__ void matrixMultiplyKernel(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float value = 0.0f;

    int tiles = (N + TILE - 1) / TILE;
    for (int t = 0; t < tiles; ++t) {
        int a_col = t * TILE + threadIdx.x; // column in A
        int b_row = t * TILE + threadIdx.y; // row in B

        // Load A tile element (coalesced across threadIdx.x)
        if (row < M && a_col < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + a_col];
        }
        else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B tile element (coalesced across threadIdx.x)
        if (b_row < N && col < K) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * K + col];
        }
        else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Unrolling
#pragma unroll
        for (int i = 0; i < TILE; ++i) {
            value += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < K) {
        C[row * K + col] = value;
    }
}

void checkCuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " -> " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int M = 4, N = 3, K = 5;
    size_t sizeA = M * N * sizeof(float);
    size_t sizeB = N * K * sizeof(float);
    size_t sizeC = M * K * sizeof(float);

    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_C = (float*)malloc(sizeC);

    for (int i = 0; i < M * N; ++i) h_A[i] = static_cast<float>(rand() % 10);
    for (int i = 0; i < N * K; ++i) h_B[i] = static_cast<float>(rand() % 10);

    float* d_A, * d_B, * d_C;
    checkCuda(cudaMalloc(&d_A, sizeA), "Allocating A");
    checkCuda(cudaMalloc(&d_B, sizeB), "Allocating B");
    checkCuda(cudaMalloc(&d_C, sizeC), "Allocating C");

    checkCuda(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice), "Copying A");
    checkCuda(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice), "Copying B");

    // blockDim matches TILE
    dim3 blockDim(16, 16);
    dim3 gridDim((K + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    // Timer setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warm-up
    matrixMultiplyKernel << <gridDim, blockDim >> > (d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    matrixMultiplyKernel << <gridDim, blockDim >> > (d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Kernel Execution Time: " << ms << " ms\n";

    checkCuda(cudaGetLastError(), "Kernel launch");
    checkCuda(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost), "Copying C back");

    // Print matrices (small sizes)
    std::cout << "Matrix A (" << M << "x" << N << "):\n";
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) std::cout << h_A[i * N + j] << " ";
        std::cout << "\n";
    }
    std::cout << "\nMatrix B (" << N << "x" << K << "):\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) std::cout << h_B[i * K + j] << " ";
        std::cout << "\n";
    }
    std::cout << "\nMatrix C (" << M << "x" << K << "):\n";
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) std::cout << h_C[i * K + j] << " ";
        std::cout << "\n";
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
