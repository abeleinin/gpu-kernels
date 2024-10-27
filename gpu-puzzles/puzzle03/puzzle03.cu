#include <cuda.h>

#include "../utils.h"

__global__ void guardKernel(float* a, float* out, int n) {
  int i = threadIdx.x;
  if (i < n) {
    out[i] = a[i] + 10;
  }
}

void guardTest(float* a, float* out, int n) {
  float *a_d, *out_d;
  int size = n * sizeof(float);

  cudaMalloc((void**) &a_d, size);
  cudaMalloc((void**) &out_d, size);

  cudaMemcpy(a_d, a, size, cudaMemcpyHostToDevice);

  guardKernel<<<1, 8>>>(a_d, out_d, n);
  
  cudaMemcpy(out, out_d, size, cudaMemcpyDeviceToHost);

  cudaFree(a_d);
  cudaFree(out_d);
}


int main(void) {
  const int n = 4;
  float a[n];
  float out[n];

  init_arr(a, n);

  guardTest(a, out, n);

  print_arr(out, n);

  return 0;
}

