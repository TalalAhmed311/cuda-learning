#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 512  // Number of rows in A and C
#define K 1024  // Number of columns in A and rows in B
#define N 512  // Number of columns in B and C
#define BLOCK_SIZE 64
#define TILE_SIZE 32

// Initialize matrix with random values
void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

void matmul_cpu(float *matA, float *matB, float *matC, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += matA[i * k + l] * matB[l * n + j];
            }
            matC[i * n + j] = sum;
        }
    }
}

__global__ void matmul_gpu(float *matA, float *matB, float *matC, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int l = 0; l < k; l++) {
            sum += matA[row * k + l] * matB[l * n + col];
        }
        matC[row * n + col] = sum;
    }
}

__global__ void matmul_tiled(float* A, float* B, float* C, int m, int n, int k) {

    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        if (row < m && tile * TILE_SIZE + tx < k)
            sharedA[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
        else
            sharedA[ty][tx] = 0.0f;

        if (col < N && tile * TILE_SIZE + ty < K)
            sharedB[ty][tx] = B[(tile * TILE_SIZE + ty) * n + col];
        else
            sharedB[ty][tx] = 0.0f;

        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += sharedA[ty][k] * sharedB[k][tx];

        __syncthreads();
    }
    
    if (row < m && col < n)
        C[row * n + col] = sum;
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    float *matA, *matB, *matC_cpu, *matC_gpu, *matC_tiled;
    float *d_A, *d_B, *d_C_gpu, *d_C_tiled;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    matA = (float*)malloc(size_A);
    matB = (float*)malloc(size_B);
    matC_cpu = (float*)malloc(size_C);
    matC_gpu = (float*)malloc(size_C);
    matC_tiled = (float*)malloc(size_C);

    srand(time(NULL));
    init_matrix(matA, M, K);
    init_matrix(matB, K, N);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C_gpu, size_C);
    cudaMalloc(&d_C_tiled, size_C);

    cudaMemcpy(d_A, matA, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, matB, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    dim3 tiledBlockDim(TILE_SIZE, TILE_SIZE);
    dim3 tiledGridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;

    for(int i=0;i<20;i++){

        double cpu_start_time = get_time();
        matmul_cpu(matA, matB, matC_cpu, M, N, K);
        double cpu_end_time = get_time();
        cpu_total_time = (cpu_end_time - cpu_start_time) * 1e6;

    }
    
    double cpu_avg_time = cpu_total_time / 20.0;


    printf("Benchmarking Naïve GPU implementation...\n");


    double gpu_total_time = 0.0f;

    for(int i=0;i<20;i++){

        double gpu_start_time = get_time();
        matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C_gpu, M, K, N);
        cudaDeviceSynchronize();
        double gpu_end_time = get_time();
        gpu_total_time = (gpu_end_time - gpu_start_time) *1e6;

    }

    double gpu_avg_time = gpu_total_time / 20.0;

 
    printf("Benchmarking Tiled GPU implementation...\n");

    double tiled_total_time = 0.0f;

    for(int i=0;i<20;i++){

        double tiled_start_time = get_time();
        matmul_tiled<<<tiledGridDim, tiledBlockDim>>>(d_A, d_B, d_C_tiled, M, N, K);
        cudaDeviceSynchronize();
        double tiled_end_time = get_time();
        tiled_total_time = (tiled_end_time - tiled_start_time) * 1e6;

    }

    double tiled_avg_time = tiled_total_time/20.0;
    
    cudaMemcpy(matC_gpu, d_C_gpu, size_C, cudaMemcpyDeviceToHost);
    cudaMemcpy(matC_tiled, d_C_tiled, size_C, cudaMemcpyDeviceToHost);

    printf("\n=== Performance Results ===\n");
    printf("CPU time: %f microseconds\n", cpu_avg_time);
    printf("GPU (Naïve) time: %f microseconds\n", gpu_avg_time);
    printf("GPU (Tiled) time: %f microseconds\n", tiled_avg_time);
    printf("Speedup (Naïve GPU vs CPU): %fx\n", cpu_avg_time / gpu_avg_time);
    printf("Speedup (Tiled GPU vs CPU): %fx\n", cpu_avg_time / tiled_avg_time);
    printf("Speedup (Tiled GPU vs Naïve GPU): %fx\n", gpu_avg_time / tiled_avg_time);

    free(matA);
    free(matB);
    free(matC_cpu);
    free(matC_gpu);
    free(matC_tiled);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_gpu);
    cudaFree(d_C_tiled);

    return 0;
}
