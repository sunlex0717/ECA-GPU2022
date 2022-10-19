#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <cmath>

#include <chrono>// timer library

//feel free to change these parameters when you do experiments
#define DATA_SIZE 51200
#define BLOCK_SIZE 256
#define MASK_WIDTH 155 // we assune the mask width should be odd value

#define TEST_ROUNDS 1000

__constant__ float mask_const[MASK_WIDTH];




void convolution_1D_host(float *in, float *m, float *out, int Mask_Width, int Width) {
    for(int i = 0; i<Width;i++){
        float Res = 0;
        int In_start_point = i - (Mask_Width/2);
        for (int j = 0; j < Mask_Width; j++) {
            if (In_start_point + j >= 0 && In_start_point + j < Width) {
                Res += in[In_start_point + j]*m[j];
            }
        }
        out[i] = Res;
    }
}  

// Naive implementation
__global__ void convolution_1D_basic_kernel(float *in, float *m, float *out, int Mask_Width, int Width) {

    int gindex = threadIdx.x + blockIdx.x * blockDim.x;  //gloabl index
    
    int In_start_point = gindex - int(Mask_Width/2);//start index of input data
    float result = 0;
    for(int i = 0; i < Mask_Width ;i++){
        if( In_start_point + i >= 0   && In_start_point + i < Width ) { // boundary check
            result += in[In_start_point+i]*m[i];
        }
    }
    out[gindex] = result;
}
  

// Tiled convolution  Implement Your self
__global__ void convolution_1D_basic_tiled_kernel(float *in, float *m, float *out, int Mask_Width, int Width) {


}

// Implement yourself, tiled with constant memory
__global__ void convolution_1D_tiled_const_kernel(float *in, float *out, int Mask_Width, int Width) {



}


void initial(float *in){
    float LO = -1;
    float HI = 1;
    for (int i = 0; i < DATA_SIZE;i++){
        in[i] = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
    }
}

void initial_mask(float *mask){
    float LO = -1;
    float HI = 1;
    for (int i =0; i < MASK_WIDTH;i++){
        mask[i] = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
    }
}

void print_mask(float* mask){
    std::cout<<"[";
    for(int i =0; i < MASK_WIDTH; i++){
        
        std::cout<<mask[i]<<",";
    }
    std::cout<<"]"<<std::endl;
}

void printOut(float *out){
    std::cout<<"[";
    for(int i =0; i < DATA_SIZE; i++){
        
        std::cout<<out[i]<<",";
    }
    std::cout<<"]"<<std::endl;
}


void check_errors(float* h_out, float* d_out, int size){
    float errors = 0;
    for(int i =0;i < size;i++){
        errors += abs(h_out[i] - d_out[i]);
    }
    float avg_err = errors/size;
    //std::cout << "average errors = " << errors/size<<std::endl;
    if(avg_err > 0.001){
        std::cout << "average errors = " << avg_err<<std::endl;
        std::cout << " error: Check your CUDA implementation! the result is not numerically correct compared to C program" << std::endl;
    }
}


void run_C(){


    float in[DATA_SIZE]; 
    float mask[MASK_WIDTH]; 
    float out[DATA_SIZE];


    //convolution_1D_host(float *in, float *m, float *out, int Mask_Width, int Width)

    initial(in);
    initial(mask);
    float total = 0;
    for(int i=0; i<TEST_ROUNDS;i++){

        auto begin = std::chrono::high_resolution_clock::now();
        convolution_1D_host(in,mask,out,MASK_WIDTH,DATA_SIZE);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<float, std::milli> fp_ms = end - begin;

        total += fp_ms.count();
    }



    std::cout << "C program time  : " << total/TEST_ROUNDS << " ms " << std::endl;



}

void run_Naive_CUDA(){
    float in[DATA_SIZE]; 
    float mask[MASK_WIDTH]; 
    float out[DATA_SIZE];

    float host_out[DATA_SIZE];

    float *d_in, *d_out, *d_mask;
    int size = sizeof(float) * DATA_SIZE;
    int size_mask = sizeof(float) * MASK_WIDTH;

    cudaMalloc( (void **) &d_in, size);
    cudaMalloc( (void **) &d_out, size);
    cudaMalloc( (void **) &d_mask, size_mask ); 
 
    // host data initialization
    initial(in);
    initial(mask);
    //print_mask(mask);
    // std::cout<<"Input = "<< std::endl;
    // printOut(in);

    // CUDA memory copy function, from host to device
    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, size_mask, cudaMemcpyHostToDevice);

    //create CUDA event to measure the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // do compuation on device(GPU)

    float total = 0;

    // do compuation on device(GPU)
    for (int i = 0;i < TEST_ROUNDS;i++ ){

        cudaEventRecord(start);

        convolution_1D_basic_kernel<<< (DATA_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(d_in,d_mask,d_out,MASK_WIDTH,DATA_SIZE);
    
        cudaDeviceSynchronize();
    
        cudaEventRecord(stop);
    
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total += milliseconds;

    }

    

    std::cout << "Naive CUDA Kernel Execution time(without data transfer time) = " << total/TEST_ROUNDS  << " ms"<< std::endl;
    
    
    // Copy result back to host (CPU)
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);
    //error check and report any possible GPU errors

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout<< " CUDA failed "<< std::endl;
        std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
    }


    convolution_1D_host(in, mask, host_out,  MASK_WIDTH,  DATA_SIZE);

    check_errors(host_out,out,DATA_SIZE);

    cudaFree(d_out);
    cudaFree(d_in);
    cudaFree(d_mask);

}


void run_tiled_CUDA(){
    float in[DATA_SIZE]; 
    float mask[MASK_WIDTH]; 
    float out[DATA_SIZE];

    float host_out[DATA_SIZE];

    float *d_in, *d_out, *d_mask;
    int size = sizeof(float) * DATA_SIZE;
    int size_mask = sizeof(float) * MASK_WIDTH;

    cudaMalloc( (void **) &d_in, size);
    cudaMalloc( (void **) &d_out, size);
    cudaMalloc( (void **) &d_mask, size_mask ); 
 
    // host data initialization
    initial(in);
    initial(mask);
    //print_mask(mask);
    // std::cout<<"Input = "<< std::endl;
    // printOut(in);

    // CUDA memory copy function, from host to device
    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, size_mask, cudaMemcpyHostToDevice);
    //create CUDA event to measure the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total = 0;

    // do compuation on device(GPU)
    for (int i = 0;i < TEST_ROUNDS;i++ ){

        cudaEventRecord(start);

        convolution_1D_basic_tiled_kernel<<< (DATA_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(d_in,d_mask,d_out,MASK_WIDTH,DATA_SIZE);
    
        cudaDeviceSynchronize();
    
        cudaEventRecord(stop);
    
    
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total += milliseconds;

    }

    

    std::cout << "Tiled Kernel Execution time(without data transfer time) = " << total/TEST_ROUNDS  << " ms"<< std::endl;

    // Copy result back to host (CPU)
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);
    //error check and report any possible GPU errors

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout<< " CUDA failed "<< std::endl;
        std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
    }

    convolution_1D_host(in, mask, host_out,  MASK_WIDTH,  DATA_SIZE);

    check_errors(host_out,out,DATA_SIZE);


    cudaFree(d_out);
    cudaFree(d_in);
    cudaFree(d_mask);

}



void run_tiled_const_CUDA(){
    float in[DATA_SIZE]; 
    float mask[MASK_WIDTH]; 
    float out[DATA_SIZE];

    float host_out[DATA_SIZE];

    float *d_in, *d_out;
    int size = sizeof(float) * DATA_SIZE;
    int size_mask = sizeof(float) * MASK_WIDTH;

    cudaMalloc( (void **) &d_in, size);
    cudaMalloc( (void **) &d_out, size);
 
    // host data initialization
    initial(in);
    initial(mask);
    //print_mask(mask);
    // std::cout<<"Input = "<< std::endl;
    // printOut(in);

    // CUDA memory copy function, from host to device
    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);

    // constant memory
    cudaMemcpyToSymbol(mask_const,mask,size_mask);

    //create CUDA event to measure the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total = 0;

    // do compuation on device(GPU)
    for (int i = 0;i < TEST_ROUNDS;i++ ){

        cudaEventRecord(start);

        convolution_1D_tiled_const_kernel<<< (DATA_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(d_in,d_out,MASK_WIDTH,DATA_SIZE); 
    
        cudaDeviceSynchronize();
    
        cudaEventRecord(stop);
    
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total += milliseconds;

    }

    std::cout << "Tiled + Const Mem CUDA Kernel Execution time(without data transfer time) = " << total/TEST_ROUNDS  << " ms" << std::endl;
    

    // Copy result back to host (CPU)
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);
    //error check and report any possible GPU errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout<< " CUDA failed "<< std::endl;
        std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
    }

    // compute C
    convolution_1D_host(in, mask, host_out,  MASK_WIDTH,  DATA_SIZE);

    check_errors(host_out,out,DATA_SIZE);

    cudaFree(d_out);
    cudaFree(d_in);


}



int main() {
    run_C();
    run_Naive_CUDA();
    run_tiled_CUDA();
    run_tiled_const_CUDA();



    return 0;
}