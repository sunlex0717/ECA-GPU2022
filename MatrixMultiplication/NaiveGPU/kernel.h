#ifndef _KERNEL_H_
#define _KERNEL_H_

#include <cuda_runtime.h>

void computeGold(float* C, const float* A, const float* B, unsigned int N ); // CPU 
__global__ void matrixMul_naive(float* A, float* B, float* C, int N);



#endif // _KERNEL_H_