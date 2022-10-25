#include "image_processing_functions_gpu.h"

__global__ void greyscale_gpu(uint8_t const * const frame_in, uint32_t const xsize_in, uint32_t const ysize_in, uint32_t const bitsperpixel_in,
                      uint8_t * const frame_out)
                      
{



}



__global__ void convolution_gpu(uint8_t const * const frame_in, uint32_t const xsize_in, uint32_t const ysize_in, uint32_t const bitsperpixel_in,
                        double const * const f, uint32_t const fxsize, uint32_t const fysize, uint8_t * const frame_out)
{

    
}