#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "image_processing_functions.h"

int main(int argc, char ** argv)
{
  if (argc < 3) {
    fprintf(stderr, "Usage: %s infile outfile\n" , argv[0]);
    return 1;
  }

  // read BMP
  uint32_t xsize1, ysize1, bitsperpixel1;
  uint32_t xsize2, ysize2, bitsperpixel2;
  uint32_t xsize3, ysize3, bitsperpixel3;
  // uint32_t xsize4, ysize4, bitsperpixel4;
  // uint32_t xsize5, ysize5, bitsperpixel5;
  uint8_t threshold;
  uint8_t * frame1 = NULL;
  if (!readBMP(argv[1], &frame1, &xsize1, &ysize1, &bitsperpixel1)) return 1;

  uint32_t bytes2, bytes3;
  uint8_t * frame2, * frame3;
  if (bitsperpixel1 == 24) {
    xsize2 = xsize1;
    ysize2 = ysize1;
    bitsperpixel2 = 8;
    bytes2 = xsize2*ysize2*(bitsperpixel2/8);
    frame2 = (uint8_t *) malloc (bytes2);
    greyscale(frame1, xsize1, ysize1, bitsperpixel1, frame2);
    threshold = 128;
    xsize3 = xsize2;
    ysize3 = ysize2;
    bitsperpixel3 = bitsperpixel2;
    bytes3 = xsize3*ysize3*(bitsperpixel3/8);
    frame3 = (uint8_t *) malloc (bytes3);
    convolution(frame2, xsize2, ysize2, bitsperpixel2, conv_avgxy3, 3, 3, frame3);
  } else {
    xsize3 = xsize1;
    ysize3 = ysize1;
    bitsperpixel3 = bitsperpixel1;
    bytes3 = xsize3*ysize3*(bitsperpixel3/8);
    frame3 = (uint8_t *) malloc (bytes3);
    convolution(frame1, xsize1, ysize1, bitsperpixel1, conv_gaussianblur5, 5, 5, frame3);
    threshold = 100;
  }
      
  // xsize4 = xsize3;
  // ysize4 = ysize3;
  // bitsperpixel4 = bitsperpixel3;
  // uint32_t const bytes4 = xsize4*ysize4*(bitsperpixel4/8);
  // uint8_t * const frame4 = (uint8_t *) malloc (bytes4);
  // sobel(frame3, xsize3, ysize3, bitsperpixel3, threshold, frame4);
  
  // xsize5 = xsize4;
  // ysize5 = ysize4;
  // bitsperpixel5 = bitsperpixel4;
  // uint32_t const bytes5 = xsize4*ysize4*(bitsperpixel4/8);
  // uint8_t * const frame5 = (uint8_t *) malloc (bytes5);
  // overlay(frame3, xsize3, ysize3, bitsperpixel3, frame4, xsize4, ysize4, bitsperpixel4, 0, 0, 0.7, frame5);

  // write BMP
  // note that reading & then writing doesn't always result in the same image
  // - a grey-scale (8-bit pixel) image will be written as a 24-bit pixel image too
  // - the header of colour images may change too
  uint32_t const r = writeBMP(argv[2], frame3, xsize3, ysize3, bitsperpixel3);
  return r;
}
