#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>

#define DATA_SIZE 120
#define BLOCK_SIZE 12
#define RADIUS 2



void initial(int *in){
    for (int i = 0; i < DATA_SIZE;i++){
        in[i] = i;
    }
}

void printOut(int *out){
    std::cout<<"[";
    for(int i =0; i < DATA_SIZE; i++){
        
        std::cout<<out[i]<<",";
    }
    std::cout<<"]"<<std::endl;;
}


  
// Naive implementation
__global__ void stencil_1d_simple(int *in, int *out) {

    int gindex = threadIdx.x + blockIdx.x * blockDim.x;  //gloabl index
    
    int In_start_point = gindex - RADIUS;//start index of input data
    int result = 0;
    for(int i = 0; i < RADIUS*2 + 1;i++){
        if( In_start_point + i >= 0   && In_start_point + i < DATA_SIZE ) { // boundary check
            result += in[In_start_point+i];
        }
    }
    out[gindex] = result;
}


// Tiling
__global__ void stencil_1d_tile(int *in, int *out) {

    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;  //gloabl index
    int lindex = threadIdx.x + RADIUS; //local index
    // Read input elements into shared memory
    temp[lindex] = in[gindex];
    // elements of halo cell
    if (threadIdx.x < RADIUS) { // we only need RADIUS threads to load elements in halo cells
      temp[lindex - RADIUS] = in[gindex - RADIUS]; // left halo cell
      temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE]; // right halo cell
    }
    __syncthreads();

    // Apply the stencil
    int result = 0;
    for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
        result += temp[lindex + offset];
    // Store the result
    out[gindex] = result;
}


int main() {
    int in[DATA_SIZE];
    int out[DATA_SIZE];


    int *d_in, *d_out;
    int size = sizeof(int) * DATA_SIZE;

    cudaMalloc( (void **) &d_in, size);
    cudaMalloc( (void **) &d_out, size);
 
    initial(in);

    // CUDA memory copy function, from host to device
    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);

    //create CUDA event to measure the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // start recording
    cudaEventRecord(start);

    //stencil_1d_tile<<< (DATA_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(d_in,d_out);

    stencil_1d<<< (DATA_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(d_in,d_out);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    // Copy result back to host (CPU)
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    //std::cout << "Kernel Execution time(without data transfer time) = " << milliseconds << std::endl;
    
    // error check and report any possible GPU errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
    }
    return 0;
}