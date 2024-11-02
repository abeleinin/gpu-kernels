#include <cuda.h>

#include "utils.h"

__global__ void zip_kernel(float* a, float* b, float* out, int n) {
  int i = threadIdx.x;
  out[i] = a[i] + b[i];
}

void zip_test(float* a, float* b, float* out, int n) {
  float *a_d, *b_d, *out_d;
  int size = n * sizeof(float);

  cudaMalloc((void**) &a_d, size);
  cudaMalloc((void**) &b_d, size);
  cudaMalloc((void**) &out_d, size);

  cudaMemcpy(a_d, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, size, cudaMemcpyHostToDevice);

  zip_kernel<<<1, 4>>>(a_d, b_d, out_d, n);
  
  cudaMemcpy(out, out_d, size, cudaMemcpyDeviceToHost);

  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(out_d);
}


int main(void) {
  int n = 4;
  float a[n];
  float b[n];
  float out[n];

  arange_array(a, n);
  arange_array(b, n);

  zip_test(a, b, out, n);

  print_array(out, n);

  return 0;
}

