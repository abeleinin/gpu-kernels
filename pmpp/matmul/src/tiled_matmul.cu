#include <cuda.h>

#include "utils.h"

#define TILE_WIDTH 16

// Figure 5.9 PMPP
// A tiled matrix multiplication kernel using shared memory.
__global__ void tiled_matmul_kernel(float* M, float* N, float* P, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;
    for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) {
        Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    P[Row * Width + Col] = Pvalue;
}

void matmul(float* Min, float* Nin, float* Pout, int width) {
  float *Min_d, *Nin_d, *Pout_d;
  int size = width * width * sizeof(float);

  cudaMalloc((void**) &Min_d, size);
  cudaMalloc((void**) &Nin_d, size);
  cudaMalloc((void**) &Pout_d, size);

  cudaMemcpy(Min_d, Min, size, cudaMemcpyHostToDevice);
  cudaMemcpy(Nin_d, Nin, size, cudaMemcpyHostToDevice);
  cudaMemcpy(Pout_d, Pout, size, cudaMemcpyHostToDevice);

  // Dynamically allocate threads dependant on image size
  dim3 dimGrid(ceil(width*width / 16.0), ceil(width*width / 16.0), 1);
  dim3 dimBlock(16, 16, 1);

  tiled_matmul_kernel<<<dimGrid, dimBlock>>>(Min_d, Nin_d, Pout_d, width);
  
  cudaMemcpy(Pout, Pout_d, size, cudaMemcpyDeviceToHost);

  cudaFree(Min_d);
  cudaFree(Nin_d);
  cudaFree(Pout_d);
}

int main(void) {
  const int n = 9;
  float A[n];
  float B[n];
  float C[n];

  // set values of input arrays
  for (int i = 0; i < n; ++i) {
    A[i] = i;
    B[i] = i;
  }

  print_array(A, n);
  print_array(B, n);

  matmul(A, B, C, 3);

  print_array(C, n);

  return 0;
}
