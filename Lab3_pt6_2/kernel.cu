#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdlib>
#include <stdio.h>

#define TILE 8            // must be multiple of 4
#define VEC 4              // how many columns each thread computes

static_assert(TILE% VEC == 0, "TILE must be divisible by VEC (4)");

__global__ void matrixMultiplyKernel_vec4(const float* A, const float* B, float* C, int M, int N, int K) {
    // shared tiles full size
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    // thread tile coords:
    int tx = threadIdx.x;          // ranges 0 .. TILE/VEC - 1
    int ty = threadIdx.y;          // ranges 0 .. TILE-1

    int block_col_base = blockIdx.x * TILE; // base column this block handles
    int block_row_base = blockIdx.y * TILE; // base row

    int row = block_row_base + ty;
    int col_base = block_col_base + tx * VEC; // this thread handles col_base + [0..3]

    float accum[VEC] = { 0.0f, 0.0f, 0.0f, 0.0f };

    int tiles = (N + TILE - 1) / TILE;
    for (int t = 0; t < tiles; ++t) {
        int a_col_base = t * TILE; // columns in A for this tile
        int b_row_base = t * TILE; // rows in B for this tile

        // Each thread loads VEC elements of A (row fixed, consecutive cols) into shared memory.
        // Compute element indices and check bounds
        for (int v = 0; v < VEC; ++v) {
            int a_col = a_col_base + tx * VEC + v; // global column in A
            if (row < M && a_col < N) {
                As[ty][tx * VEC + v] = A[row * N + a_col];
            }
            else {
                As[ty][tx * VEC + v] = 0.0f;
            }
        }

        // Each thread loads VEC elements of B: consecutive columns in the tile's row
        // B indexing: (b_row) * K + (block_col_base + tx*VEC + v)
        for (int v = 0; v < VEC; ++v) {
            int b_row = b_row_base + ty;
            int b_col = block_col_base + tx * VEC + v;
            if (b_row < N && b_col < K) {
                Bs[ty][tx * VEC + v] = B[b_row * K + b_col];
            }
            else {
                Bs[ty][tx * VEC + v] = 0.0f;
            }
        }

        __syncthreads();

        // Multiply accumulate - unrolled
#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            float a_val0 = As[ty][k];
            // read the 4 B values at [k][tx*VEC + v]
            float b0 = Bs[k][tx * VEC + 0];
            float b1 = Bs[k][tx * VEC + 1];
            float b2 = Bs[k][tx * VEC + 2];
            float b3 = Bs[k][tx * VEC + 3];
            accum[0] += a_val0 * b0;
            accum[1] += a_val0 * b1;
            accum[2] += a_val0 * b2;
            accum[3] += a_val0 * b3;
        }

        __syncthreads();
    }

    // Write back results (each thread writes up to 4 outputs)
    for (int v = 0; v < VEC; ++v) {
        int out_row = row;
        int out_col = col_base + v;
        if (out_row < M && out_col < K) {
            C[out_row * K + out_col] = accum[v];
        }
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

    // blockDim: TILE/VEC by TILE
    dim3 blockDim(TILE / VEC, TILE);                 // e.g. 8 x 32 if TILE=32,VEC=4
    dim3 gridDim((K + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    // Timer setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warm-up
    matrixMultiplyKernel_vec4 << <gridDim, blockDim >> > (d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    matrixMultiplyKernel_vec4 << <gridDim, blockDim >> > (d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Kernel Execution Time: " << ms << " ms\n";

    checkCuda(cudaGetLastError(), "Kernel launch");
    checkCuda(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost), "Copying C back");

    // Print matrices
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
