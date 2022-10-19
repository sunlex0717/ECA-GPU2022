#include <iostream>
#include <vector>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include "Matmul.h"
#include "kernel.h"
using namespace std;

// host code
void printDiff(float *data1, float *data2, int width, int height)
{
  int i,j,k;
  int error_count=0;
  for (j=0; j<height; j++) {
    for (i=0; i<width; i++) {
      k = j*width+i;
      if (fabs(data1[k] - data2[k]) > 0.1 ) {
         printf("diff(%d,%d) CPU=%4.4f, GPU=%4.4f \n", i,j, data1[k], data2[k]);
         error_count++;
      }
    }
  }
  printf("Total Errors = %d \n", error_count);
}

int main(){
    // Perform matrix multiplication C = A*B
    // where A, B and C are NxN matrices
   int N = MatDim;
   int SIZE = N*N;
   size_t mem_size = sizeof(float) * SIZE;
   std::cout << "Matrix Size : "  << N << " x " << N <<std::endl; 
   // Allocate memory on the host
   float* h_A;
   h_A = new float[SIZE];
   float* h_B;
   h_B = new float[SIZE];
   float* h_C;
   h_C = new float[SIZE];
   // set seed for rand()
   srand(2019);
   // Initialize matrices on the host with random values
    for (int i=0; i<N; i++){
      for (int j=0; j<N; j++){
          h_A[i*N+j] = rand() / (float)RAND_MAX;
          h_B[i*N+j] = rand() / (float)RAND_MAX;
      }
    }

    /**** error check ******/
    int devID = 0;

    cudaError_t error;
    cudaDeviceProp deviceProp;
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
    }

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        exit(EXIT_SUCCESS);
    }

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
    }
    else
    {
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }
    /****************end of error checks****************/

    // utilities
    cudaEvent_t start; // to record processing time
    cudaEvent_t stop;
    float msecTotal;


    /**************** allocate device memory****************/
    // create a pointer for device memory A
    float* d_A;//Mat A
    // allocate a memory sace with SIZE*sizeof(float) for device
    cudaMalloc( &d_A, mem_size);

    float* d_B;//Mat B
    cudaMalloc( &d_B, mem_size);

    float* d_C;//Mat C
    cudaMalloc( &d_C, mem_size);

    /****************** end of allocate device memory***********/

#if RUN_CPU == 1
    /***********************compute results on CPU***/
    std::cout << "Start CPU processing" << std::endl;
    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL); 
    // compute reference solution
    float* reference;
    reference = new float[SIZE];
    computeGold(reference, h_A, h_B, N);
    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    float cpu_time = msecTotal;
    std::cout <<"Naive CPU processing time: " << cpu_time << " ms" <<std::endl;

    
#endif    
    /*********Codes for Naive GPU**************************/
    // setup execution parameters
    std::cout << "start Naive GPU processing" <<std::endl;
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid(1, 1);
    blocksPerGrid.x = ceil(double(N)/double(threadsPerBlock.x));
    blocksPerGrid.y = ceil(double(N)/double(threadsPerBlock.y));
    std::cout << "block Dim  " << threadsPerBlock.x << " x "  << threadsPerBlock.y <<std::endl;
    std::cout << "Grid Dim  " << blocksPerGrid.x << " x "  << blocksPerGrid.y <<std::endl;

    // create and start timer
    // note  memory access time 
    
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL);
    // copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size,
                              cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size,
                              cudaMemcpyHostToDevice);

    /***** invoke kernel****/
    /***** replace the kernel you want***/
    matrixMul_naive<<<blocksPerGrid,threadsPerBlock>>>( d_A,d_B,d_C, N);
    // copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size,cudaMemcpyDeviceToHost);
    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    float gpu_naive_time = msecTotal;
    std::cout <<"CUDA kernel processing time: " << gpu_naive_time << " ms" <<std::endl;
    std::cout <<"speed-up : time_CPU/time_GPU " << cpu_time/gpu_naive_time <<std::endl;

    /*********Codes for Tiling GPU**************************/


    std::cout << "start GPU improvement processing" <<std::endl;
    blocksPerGrid.x = ceil(double(N)/double(threadsPerBlock.x));
    blocksPerGrid.y = ceil(double(N)/double(threadsPerBlock.y));
    std::cout << "block Dim  " << threadsPerBlock.x << " x "  << threadsPerBlock.y <<std::endl;
    std::cout << "Grid Dim  " << blocksPerGrid.x << " x "  << blocksPerGrid.y <<std::endl;

    // create and start timer
    // note  memory access time 
    
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL);
    // copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size,
                              cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size,
                              cudaMemcpyHostToDevice);

    /***** invoke kernel****/
    /***** replace the kernel you want***/
    matrixMul_coalescing<<<blocksPerGrid,threadsPerBlock>>>( d_C, d_A,d_B,N,N);
    // copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size,cudaMemcpyDeviceToHost);
    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    float gpu_improve_time = msecTotal;
    std::cout <<"CUDA kernel processing time: " << gpu_improve_time << " ms" <<std::endl;
    std::cout <<"speed-up : time_CPU/time_GPU_Improvement " << cpu_time/gpu_improve_time <<std::endl;
    std::cout <<"speed-up : time_GPU_Naive/time_GPU_Improvement " << gpu_naive_time/gpu_improve_time <<std::endl;


#if RUN_CPU == 1   
    printDiff(h_C,reference,N,N);
    
#endif
    // clean up memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;



    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

