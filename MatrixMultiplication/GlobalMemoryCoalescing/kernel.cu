/* Matrix multiplication: C = A * B.
 * Device code.
 */
 
 #include <stdio.h>
 #include "Matmul.h"

 
 #define CHECK_BANK_CONFLICTS 0
 #if CHECK_BANK_CONFLICTS
 #define AS(i, j) cutilBankChecker(((float*)&As[0][0]), (BLOCK_SIZE * i + j))
 #define BS(i, j) cutilBankChecker(((float*)&Bs[0][0]), (BLOCK_SIZE * i + j))
 #else
 #define AS(i, j) As[i][j]
 #define BS(i, j) Bs[i][j]
 #endif



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
 



 
 ////////////////////////////////////////////////////////////////////////////////
 //! Matrix multiplication on the device: C = A * B
 //! wA is A's width and wB is B's width
 ////////////////////////////////////////////////////////////////////////////////
 __global__ void
 matrixMul_coalescing( float* C, float* A, float* B, int wA, int wB)
 {
     // Block index
     int bx = blockIdx.x;
     int by = blockIdx.y;
 
     // Thread index
     int tx = threadIdx.x;
     int ty = threadIdx.y;
 
     // Declaration of the shared memory array As used to
     // store the sub-matrix of A
     __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
 
     // Declaration of the shared memory array Bs used to
     // store the sub-matrix of B
     __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
 
     // Index of the first sub-matrix of A processed by the block
     int aBegin = wA * BLOCK_SIZE * by;
 
     // Index of the last sub-matrix of A processed by the block
     int aEnd   = aBegin + wA - 1;
 
     // Step size used to iterate through the sub-matrices of A
     int aStep  = BLOCK_SIZE;
 
     // Index of the first sub-matrix of B processed by the block
     int bBegin = BLOCK_SIZE * bx;
 
     // Step size used to iterate through the sub-matrices of B
     int bStep  = BLOCK_SIZE * wB;
 
     // Csub is used to store the element of the block sub-matrix
     // that is computed by the thread
     float Csub = 0;
 
     // Loop over all the sub-matrices of A and B
     // required to compute the block sub-matrix
     for (int a = aBegin, b = bBegin;
              a <= aEnd;
              a += aStep, b += bStep) {
 
 
         // Load the matrices from device memory
         // to shared memory; each thread loads
         // one element of each matrix
         AS(ty, tx) = A[a + wA * ty + tx];
         BS(tx, ty) = B[b + wB * ty + tx];
 
         // Synchronize to make sure the matrices are loaded
         __syncthreads();
 
         // Multiply the two matrices together;
         // each thread computes one element
         // of the block sub-matrix
         for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += AS(ty, k) * BS(tx, k);
 
         // Synchronize to make sure that the preceding
         // computation is done before loading two new
         // sub-matrices of A and B in the next iteration
         __syncthreads();
     }
 
     // Write the block sub-matrix to device memory;
     // each thread writes one element
     int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
     C[c + wB * ty + tx] = Csub;
 }
 
 