#pragma once
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdlib.h>
#include <iostream>

uint32_t readBMP(char const * const file, uint8_t ** frame_out, uint32_t * const xsize_in, uint32_t * const ysize_in, uint32_t * const bitsperpixel);
uint32_t writeBMP(char const * const file, uint8_t const * const frame_out, uint32_t const xsize_in, uint32_t const ysize_in, uint32_t const bitsperpixel);

void greyscale(uint8_t const * const frame_in, uint32_t const xsize_in, uint32_t const ysize_in, uint32_t const bitsperpixel_in, uint8_t * const frame_out);

double const conv_sharpen3[] = { 
  -1, -1, -1, 
  -1, 9, -1, 
  -1, -1, -1, 
};
double const conv_avgxy1[] = { 1 };
double const conv_avgx3[] = { 0.33, 0.33, 0.33, };
double const conv_avgxy3[] = {
  0.11, 0.11, 0.11, 
  0.11, 0.11, 0.11, 
  0.11, 0.11, 0.11, 
};
double const conv_avgxy7[] = {
  0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
  0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
  0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
  0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
  0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
  0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
  0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
};
double const conv_gaussianblur5[] = {
  1/256.0, 4/256.0, 6/256.0, 4/256.0,1/256.0,
  4/256.0,16/256.0,24/256.0,16/256.0,4/256.0,
  6/256.0,24/256.0,36/256.0,24/256.0,6/256.0,
  4/256.0,16/256.0,24/256.0,16/256.0,4/256.0,
  1/256.0, 4/256.0, 6/256.0, 4/256.0,1/256.0,
};


void convolution(uint8_t const * const frame_in, uint32_t const xsize_in, uint32_t const ysize_in, uint32_t const bitsperpixel_in, double const * const f, uint32_t const fxsize, uint32_t const fysize, uint8_t * const frame_out);
// extern void scale(uint8_t const * const frame_in, uint32_t const xsize_in, uint32_t const ysize_in, uint32_t const bitsperpixel_in,
//                   uint32_t const xsize_out, uint32_t const ysize_out, uint8_t * const frame_out);

// extern void overlay(uint8_t const * const frame_in1, uint32_t const xsize_in1, uint32_t const ysize_in1, uint32_t const bitsperpixel_in1,
//                     uint8_t const * const frame_in2, uint32_t const xsize_in2, uint32_t const ysize_in2, uint32_t const bitsperpixel_in2,
//                     uint32_t const xoffset, uint32_t yoffset, double const ratio, uint8_t * const frame_out);
// extern void sobel(uint8_t const * const frame_in, uint32_t const xsize_in, uint32_t const ysize_in, uint32_t const bitsperpixel_in,
//                   uint8_t const threshold, uint8_t * const frame_out);
// extern void overlay_sobel(uint8_t const * const frame_in, uint32_t const xsize_in, uint32_t const ysize_in, uint32_t const bitsperpixel_in,
//                           uint8_t const threshold, uint8_t * const frame_out);

