#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include "Matmul.h"
#include "kernel.h"


/*********Nvidia reference for mat mul*******/
// This function is used to compute true result of C = A*B on CPU

void computeGold(float* C, const float* A, const float* B, unsigned int N )
{
    for (unsigned int i = 0; i < N; ++i)
        for (unsigned int j = 0; j < N; ++j) {
            float sum = 0;
            for (unsigned int k = 0; k < N; ++k) {
                float a = A[i * N + k];
                float b = B[k * N + j];
                sum += a * b;
            }
            C[i * N + j] = (float)sum;
        }
}
/*********************************************************/


/*
Naive CUDA implementation for C=A*B
*/
__global__ void matrixMul_naive(float* A, float* B, float* C, int N)
{
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Accumulate row i of A and column j of B
  int ROW = by * blockDim.y + ty;
  int COL = bx * blockDim.x + tx;

  float accu = 0.0;
  
  for(int k=0; k<N; k++){
    accu = accu + A[ ROW * N + k ] * B[ k * N + COL ];
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  C[ ROW * N + COL ] = accu;

}

