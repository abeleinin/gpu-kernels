#include <cuda.h>

#include "utils.h"

#define SIZE 2

__global__ void broadcast_kernel(float* a, float* b, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < n && j < n) {
    int index = j*n + i;
    out[index] = a[i] + b[j];
  }
}

void broadcast_test(float* a, float* b, float* out) {
  float *a_d, *b_d, *out_d;
  int input_size = SIZE * sizeof(float);
  int output_size = SIZE * SIZE * sizeof(float);

  cudaMalloc((void**) &a_d, input_size);
  cudaMalloc((void**) &b_d, input_size);
  cudaMalloc((void**) &out_d, output_size);

  cudaMemcpy(a_d, a, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, a, input_size, cudaMemcpyHostToDevice);

  dim3 dimGrid(3, 3, 1);
  broadcast_kernel<<<dimGrid, 1>>>(a_d, b_d, out_d, SIZE);
  
  cudaMemcpy(out, out_d, output_size, cudaMemcpyDeviceToHost);

  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(out_d);
}

int main(void) {
  float a[SIZE] = {0, 1};
  float b[SIZE] = {0, 1};
  float out[SIZE*SIZE] = {0};

  broadcast_test(a, b, out);

  print_array(out, SIZE*SIZE);

  return 0;
}
