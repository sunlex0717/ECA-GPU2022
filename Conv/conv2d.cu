#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <cmath>

#include <chrono>// timer library



#define IN_SIZE 7
#define MASK_SIZE  3 // or 3 or 7

#define TEST_ROUNDS 1000

//#define BLOCK_SIZE 256

__constant__ float mask_const[MASK_SIZE*MASK_SIZE];



// C implementation
void conv2d_host(float* input,float* mask,float* output,int in_size,int mask_size){

    for (int out_h = 0;out_h<in_size;out_h++ ){
        for(int out_w = 0; out_w < in_size; out_w++ ){
            float res = 0;
            for(int mask_h = 0; mask_h < mask_size;mask_h++){
                for(int mask_w = 0; mask_w < mask_size;mask_w++){
                    
                    int input_h = out_h - mask_size/2 + mask_h;
                    int input_w = out_w - mask_size/2 + mask_w;
                    
                    if(input_h  >= 0   && input_h  < in_size  && input_w >=0 && input_w < in_size ){
                        res += mask[mask_w + mask_h * mask_size] * input[input_h*in_size + input_w];
                    }
                }
            }
            output[out_w + out_h*in_size] = res;
        }
    }
}


/*****************You CUDA kernel implementation*************************/


//__global__ void  conv2d_Naive_CUDA()


//__global__ void  conv2d_tiled_constant_mem()


/*****************You CUDA kernel implementation*************************/












void initial_input(float* input, int in_size){
    float LO = -1;
    float HI = 1;
    for(int h = 0; h < in_size;h++){
        for(int w = 0; w< in_size;w++){
            input[h*in_size + w] =  LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
        }
    }
}

void initial_mask(float* mask, int mask_size){
    float LO = -1;
    float HI = 1;
    for (int mask_h = 0; mask_h < mask_size; mask_h++){
        for (int mask_w = 0; mask_w < mask_size; mask_w++){
            mask[mask_h*mask_size + mask_w] = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
        }
    }
}


void print(float* in, int size){

    std::cout<<"[";
    for(int i =0; i < size; i++){
        for(int  j = 0; j < size; j++){
            std::cout<<in[j + i*size]<<",";
        }
        std::cout<<std::endl;
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





void run_c(){

    int in_size = IN_SIZE;
    int mask_size = MASK_SIZE;
    float input[in_size*in_size];

    float mask[mask_size*mask_size];

    float output[in_size*in_size];

    initial_input(input,in_size);
    initial_mask(mask,mask_size);

    float total = 0;

    for(int i=0;i<TEST_ROUNDS;i++){
        auto begin = std::chrono::high_resolution_clock::now();
        conv2d_host(input,mask,output,in_size,mask_size);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> fp_ms = end - begin;

        total += fp_ms.count();


    }

    std::cout << "C program time  : " << total/TEST_ROUNDS << " ms " << std::endl;
}



void run_Naive_CUDA(){

    int in_size = IN_SIZE;
    int mask_size = MASK_SIZE;


    float h_input[in_size*in_size];

    float h_mask[mask_size*mask_size];

    float h_output[in_size*in_size];

    float ref_out[in_size*in_size];

    float *d_in, *d_out, *d_mask;
    int size = sizeof(float) * in_size*in_size;
    int size_mask = sizeof(float) * mask_size*mask_size;


    cudaMalloc( (void **) &d_in, size);
    cudaMalloc( (void **) &d_out, size);
    cudaMalloc( (void **) &d_mask, size_mask ); 


    
    initial_input(h_input,in_size);
    initial_mask(h_mask,mask_size);


    // CUDA memory copy function, from host to device
    cudaMemcpy(d_in, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, size_mask, cudaMemcpyHostToDevice);

    //create CUDA event to measure the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total = 0;


    for (int i = 0;i < TEST_ROUNDS;i++ ){

        cudaEventRecord(start);

        //invoke your kernel  conv2d_Naive_CUDA
    
        cudaDeviceSynchronize();
    
        cudaEventRecord(stop);
    
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total += milliseconds;

    }



    std::cout << "Naive CUDA Kernel Execution time(without data transfer time) = " << total/TEST_ROUNDS  << " ms"<< std::endl;
    
    
    // Copy result back to host (CPU)
    cudaMemcpy(h_output, d_out, size, cudaMemcpyDeviceToHost);
    //error check and report any possible GPU errors

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout<< " CUDA failed "<< std::endl;
        std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
    }

    // void conv2d_host(float* input,float* mask,float* output,int in_size,int mask_size)
    conv2d_host(h_input, h_mask, ref_out,  in_size,  mask_size);

    check_errors(ref_out,h_output,in_size*in_size);

    cudaFree(d_out);
    cudaFree(d_in);
    cudaFree(d_mask);


}


void run_tiled_constant_mem(){

    int in_size = IN_SIZE;
    int mask_size = MASK_SIZE;


    float h_input[in_size*in_size];

    float h_mask[mask_size*mask_size];

    float h_output[in_size*in_size];

    float ref_out[in_size*in_size];

    float *d_in, *d_out;
    int size = sizeof(float) * in_size*in_size;
    int size_mask = sizeof(float) * mask_size*mask_size;


    cudaMalloc( (void **) &d_in, size);
    cudaMalloc( (void **) &d_out, size);
    //cudaMalloc( (void **) &d_mask, size_mask ); 


    
    initial_input(h_input,in_size);
    initial_mask(h_mask,mask_size);


    


    // CUDA memory copy function, from host to device
    cudaMemcpy(d_in, h_input, size, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_mask, h_mask, size_mask, cudaMemcpyHostToDevice);


    // constant memory
    cudaMemcpyToSymbol(mask_const,h_mask,size_mask);

    //create CUDA event to measure the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total = 0;


    for (int i = 0;i < TEST_ROUNDS;i++ ){

        cudaEventRecord(start);

        //invoke your kernel  conv2d_Naive_CUDA
    
        cudaDeviceSynchronize();
    
        cudaEventRecord(stop);
    
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total += milliseconds;

    }



    std::cout << "Tiled + Const Mem CUDA Kernel Execution time(without data transfer time) = " << total/TEST_ROUNDS  << " ms"<< std::endl;
    
    
    // Copy result back to host (CPU)
    cudaMemcpy(h_output, d_out, size, cudaMemcpyDeviceToHost);
    //error check and report any possible GPU errors

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout<< " CUDA failed "<< std::endl;
        std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
    }

    // void conv2d_host(float* input,float* mask,float* output,int in_size,int mask_size)
    conv2d_host(h_input, h_mask, ref_out,  in_size,  mask_size);

    check_errors(ref_out,h_output,in_size*in_size);

    cudaFree(d_out);
    cudaFree(d_in);
    // cudaFree(d_mask);

}


int main() {

    run_c();
    run_Naive_CUDA();
    run_tiled_constant_mem();

/************* Your CUDA program, you can follow the steps *******************/

/*************step 1 : create host/device arrary pointer********************/

/**************step 2 : allocate host/device memory ***********************/

/**************step 3 set up GPU event timer **************************/


/**************step 4 Kernel configurations & kernel launch **********************/


/**************step 5: check the GPU results with CPU baseline ****************/


/**************step 6: report GPU kernel time & free the allocated memory ****************/


    return 0;
}