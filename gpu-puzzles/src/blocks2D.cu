#include <cuda.h>

#include "utils.h"

#define N 5
#define SIZE 25

__global__ void map_block_2D_kernel(float* a, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < n && j < n) {
    int index = j*n + i;
    out[index] = a[index] + 10;
  }
}

void map_block_2D_test(float* a, float* out, int n) {
  float *a_d, *out_d;
  int size = n * n * sizeof(float);

  cudaMalloc((void**) &a_d, size);
  cudaMalloc((void**) &out_d, size);

  cudaMemcpy(a_d, a, size, cudaMemcpyHostToDevice);

  dim3 dimGrid(3, 3, 1);
  dim3 dimBlock(2, 2, 1);
  map_block_2D_kernel<<<dimGrid, dimBlock>>>(a_d, out_d, n);
  
  cudaMemcpy(out, out_d, size, cudaMemcpyDeviceToHost);

  cudaFree(a_d);
  cudaFree(out_d);
}


int main(void) {
  float a[N][N];
  float out[N][N];

  arange_array((float*)a, SIZE);

  map_block_2D_test((float*)a, (float*)out, N);

  print_array((float*)out, SIZE);

  return 0;
}
