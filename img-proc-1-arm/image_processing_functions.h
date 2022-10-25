#ifndef __IMAGE_PROCESSING_FUNCTIONS_H__
#define __IMAGE_PROCESSING_FUNCTIONS_H__

#include <stdlib.h>

extern uint32_t readBMP(char const * const file, uint8_t ** frame_out, uint32_t * const xsize_in, uint32_t * const ysize_in, uint32_t * const bitsperpixel);
extern uint32_t writeBMP(char const * const file, uint8_t const * const frame_out, uint32_t const xsize_in, uint32_t const ysize_in, uint32_t const bitsperpixel);

extern void greyscale(uint8_t const * const frame_in, uint32_t const xsize_in, uint32_t const ysize_in, uint32_t const bitsperpixel_in,
                      uint8_t * const frame_out);
extern double const conv_sharpen3[];
extern double const conv_avgxy1[];
extern double const conv_avgxy3[];
extern double const conv_avgx3[];
extern double const conv_avgxy7[];
extern double const conv_gaussianblur5[];
extern void convolution(uint8_t const * const frame_in, uint32_t const xsize_in, uint32_t const ysize_in, uint32_t const bitsperpixel_in,
                        double const * const f, uint32_t const fxsize, uint32_t const fysize, uint8_t * const frame_out);
// extern void scale(uint8_t const * const frame_in, uint32_t const xsize_in, uint32_t const ysize_in, uint32_t const bitsperpixel_in,
//                   uint32_t const xsize_out, uint32_t const ysize_out, uint8_t * const frame_out);

// extern void overlay(uint8_t const * const frame_in1, uint32_t const xsize_in1, uint32_t const ysize_in1, uint32_t const bitsperpixel_in1,
//                     uint8_t const * const frame_in2, uint32_t const xsize_in2, uint32_t const ysize_in2, uint32_t const bitsperpixel_in2,
//                     uint32_t const xoffset, uint32_t yoffset, double const ratio, uint8_t * const frame_out);
// extern void sobel(uint8_t const * const frame_in, uint32_t const xsize_in, uint32_t const ysize_in, uint32_t const bitsperpixel_in,
//                   uint8_t const threshold, uint8_t * const frame_out);
// extern void overlay_sobel(uint8_t const * const frame_in, uint32_t const xsize_in, uint32_t const ysize_in, uint32_t const bitsperpixel_in,
//                           uint8_t const threshold, uint8_t * const frame_out);

#endif
