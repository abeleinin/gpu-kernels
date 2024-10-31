#include <cuda.h>

#include "../utils.h"

__global__ void map2DKernel(float* a, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < n && j < n) {
    int index = j*n + i;
    out[index] = a[index] + 10;
  }
}

void map2DTest(float* a, float* out, int n) {
  float *a_d, *out_d;
  int size = n * n * sizeof(float);

  cudaMalloc((void**) &a_d, size);
  cudaMalloc((void**) &out_d, size);

  cudaMemcpy(a_d, a, size, cudaMemcpyHostToDevice);

  dim3 dimGrid(3, 3, 1);
  dim3 dimBlock(1, 1, 1);
  map2DKernel<<<dimGrid, dimBlock>>>(a_d, out_d, n);
  
  cudaMemcpy(out, out_d, size, cudaMemcpyDeviceToHost);

  cudaFree(a_d);
  cudaFree(out_d);
}


int main(void) {
  const int N = 2;
  const int SIZE = 4;
  
  float a[N][N];
  float out[N][N];

  arange_arr((float*)a, SIZE);

  map2DTest((float*)a, (float*)out, N);

  print_arr((float*)out, SIZE);

  return 0;
}

