#include <cuda.h>
#include <stdio.h>

// compute the vector addition of arrays A and B 
// and write the output to the vector C:
//                C = A + B
// each thread executes a single addition op
__global__ void vecAddKernel(float* A, float* B, float* C, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n) {
    C[i] = A[i] + B[i];
  }
}

// allocate memory for vecAddKernel and manage 
// data transfer between host and device
void vecAdd(float* A, float* B, float* C, int n) {
  float *A_d, *B_d, *C_d;
  int size = n * sizeof(float);

  cudaMalloc((void**) &A_d, size);
  cudaMalloc((void**) &B_d, size);
  cudaMalloc((void**) &C_d, size);

  cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

  vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);
  
  cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

// print the given array
void printArray(float* arr, int size) {
  for (int i = 0; i < size; ++i) {
    if (i > 0) {
      printf(", ");
      if (i % 10 == 0) {
        printf("\n");
      }
    }
    printf("%0.1f", arr[i]);
  }
  printf("\n"); 
}

int main(void) {
  const int n = 1000;
  float A[n];
  float B[n];
  float C[n];

  // set values of input arrays
  for (int i = 0; i < n; ++i) {
    A[i] = 1;
    B[i] = 2;
  }

  vecAdd(A, B, C, n);

  printArray(C, n);

  return 0;
}

