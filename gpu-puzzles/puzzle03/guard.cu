#include <cuda.h>

#include "../utils.h"

__global__ void guard_kernel(float* a, float* out, int n) {
  int i = threadIdx.x;
  if (i < n) {
    out[i] = a[i] + 10;
  }
}

void guard_test(float* a, float* out, int n) {
  float *a_d, *out_d;
  int size = n * sizeof(float);

  cudaMalloc((void**) &a_d, size);
  cudaMalloc((void**) &out_d, size);

  cudaMemcpy(a_d, a, size, cudaMemcpyHostToDevice);

  guard_kernel<<<1, 8>>>(a_d, out_d, n);
  
  cudaMemcpy(out, out_d, size, cudaMemcpyDeviceToHost);

  cudaFree(a_d);
  cudaFree(out_d);
}


int main(void) {
  int n = 4;
  float a[n];
  float out[n];

  arange_array(a, n);

  guard_test(a, out, n);

  print_array(out, n);

  return 0;
}

