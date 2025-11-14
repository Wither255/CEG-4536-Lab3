
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdlib>
#include <stdio.h>

#define TILE 32
//We wanna loop over Tiles of 32 as 32*8=128Bites.


__global__ void matrixMultiplyKernel(float* A, float* B, float* C, int M, int N, int K) {
    // Shared memory tiles
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float value = 0.0f;

    // Loop over tiles of size 32x32
    for (int t = 0; t < (N + TILE - 1) / TILE; t++) {

        // Each warp loads a full row = 32 consecutive floats = 128 bytes
        if (row < M && (t * TILE + threadIdx.x) < N) {
            As[threadIdx.y][threadIdx.x] =
                A[row * N + t * TILE + threadIdx.x];
        }
        else {
            As[threadIdx.y][threadIdx.x] = 0;
        }

        // Each warp loads a full row of B's tile
        if (col < K && (t * TILE + threadIdx.y) < N) {
            Bs[threadIdx.y][threadIdx.x] =
                B[(t * TILE + threadIdx.y) * K + col];
        }
        else {
            Bs[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // Compute using the tile
        for (int i = 0; i < TILE; i++) {
            value += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < K) {
        C[row * K + col] = value;
    }
}

// Helper to check for CUDA errors
void checkCuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " -> " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Matrix dimensions
    int M = 4;  // Rows of A
    int N = 3;  // Cols of A and Rows of B
    int K = 5;  // Cols of B

    size_t sizeA = M * N * sizeof(float);
    size_t sizeB = N * K * sizeof(float);
    size_t sizeC = M * K * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_C = (float*)malloc(sizeC);

    for (int i = 0; i < M * N; i++) h_A[i] = static_cast<float>(rand() % 10);
    for (int i = 0; i < N * K; i++) h_B[i] = static_cast<float>(rand() % 10);

    // Allocate CUDA memory
    float* d_A, * d_B, * d_C;
    checkCuda(cudaMalloc(&d_A, sizeA), "Allocating A");
    checkCuda(cudaMalloc(&d_B, sizeB), "Allocating B");
    checkCuda(cudaMalloc(&d_C, sizeC), "Allocating C");

    checkCuda(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice), "Copying A");
    checkCuda(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice), "Copying B");

    // CUDA grid/block size
    dim3 blockDim(16, 16);
    dim3 gridDim((K + blockDim.x - 1) / blockDim.x,
        (M + blockDim.y - 1) / blockDim.y);

   
    matrixMultiplyKernel << <gridDim, blockDim >> > (d_A, d_B, d_C, M, N, K);

   checkCuda(cudaGetLastError(), "Kernel launch");

   checkCuda(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost), "Copying C back");
    
    // Print 
    std::cout << "Matrix A (" << M << "x" << N << "):\n";
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) std::cout << h_A[i * N + j] << " ";
        std::cout << "\n";
    }

    std::cout << "\nMatrix B (" << N << "x" << K << "):\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) std::cout << h_B[i * K + j] << " ";
        std::cout << "\n";
    }

    std::cout << "\nMatrix C = A * B (" << M << "x" << K << "):\n";
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) std::cout << h_C[i * K + j] << " ";
        std::cout << "\n";
    }

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
