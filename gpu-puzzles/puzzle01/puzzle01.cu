#include <cuda.h>

#include "../utils.h"

__global__ void mapKernel(float* a, float* out, int n) {
  int i = threadIdx.x;
  out[i] = a[i] + 10;
}

void mapTest(float* a, float* out, int n) {
  float *a_d, *out_d;
  int size = n * sizeof(float);

  cudaMalloc((void**) &a_d, size);
  cudaMalloc((void**) &out_d, size);

  cudaMemcpy(a_d, a, size, cudaMemcpyHostToDevice);

  mapKernel<<<1, 4>>>(a_d, out_d, n);
  
  cudaMemcpy(out, out_d, size, cudaMemcpyDeviceToHost);

  cudaFree(a_d);
  cudaFree(out_d);
}


int main(void) {
  const int n = 4;
  float a[n];
  float out[n];

  init_arr(a, n);

  mapTest(a, out, n);

  print_arr(out, n);

  return 0;
}

