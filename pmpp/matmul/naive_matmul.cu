#include <cuda.h>
#include <stdio.h>

// Figure 3.11 from PMPP
// A matrix muliplication kernel using one thread to compute one P element.
__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < Width) && (col < Width)) {
        float Pvalue = 0;
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[row*Width+k] * N[k*Width+col];
        }
        P[row*Width+col] = Pvalue;
    }
}

// PMPP Exercise 1a
__global__ void matmul_exercise_1a(float *M, float *N, float *P, int Width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < Width) {
        for (int p = 0; p < Width; p++) {
            P[row * P + p] = 0.0f;
        }

        for (int n = 0; n < Width; n++) {
            for (int p = 0; p < Width; p++) {
                P[row * Width + p] += M[row * Width + n] * N[n * Width + p];
            }
        }
    }
}

// allocate memory for matmul and manage data transfer between host and device
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

  MatrixMulKernel<<<dimGrid, dimBlock>>>(Min_d, Nin_d, Pout_d, width);
  
  cudaMemcpy(Pout, Pout_d, size, cudaMemcpyDeviceToHost);

  cudaFree(Min_d);
  cudaFree(Nin_d);
  cudaFree(Pout_d);
}

// print the given array
void print_array(float* arr, int size) {
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
