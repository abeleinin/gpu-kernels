#include <cuda.h>

#include "utils.h"

__global__ void map_block_kernel(float* a, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + 10;
  }
}

void map_block_test(float* a, float* out, int n) {
  float *a_d, *out_d;
  int size = n * sizeof(float);

  cudaMalloc((void**) &a_d, size);
  cudaMalloc((void**) &out_d, size);

  cudaMemcpy(a_d, a, size, cudaMemcpyHostToDevice);

  map_block_kernel<<<4, 3>>>(a_d, out_d, n);
  
  cudaMemcpy(out, out_d, size, cudaMemcpyDeviceToHost);

  cudaFree(a_d);
  cudaFree(out_d);
}


int main(void) {
  const int n = 9;
  float a[n];
  float out[n];

  arange_array(a, n);

  map_block_test(a, out, n);

  print_array(out, n);

  return 0;
}
