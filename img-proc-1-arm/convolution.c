#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

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

//convolution(frame2, xsize2, ysize2, bitsperpixel2, conv_avgxy3, 3, 3, frame3);
void convolution(uint8_t const * const frame_in, uint32_t const xsize_in, uint32_t const ysize_in, uint32_t const bitsperpixel_in,
                 double const * const f, uint32_t const fxsize, uint32_t const fysize, uint8_t * const frame_out)
{
  uint32_t const bytes = bitsperpixel_in/8;
  for (uint32_t y = 0; y < ysize_in; y++) {
    for (uint32_t x = 0; x < xsize_in; x++) {
      for (uint32_t b = 0; b < bytes; b++) {
        double r = 0;
        for (uint32_t ty = 0; ty < fysize; ty++) {
          for (uint32_t tx = 0; tx < fxsize; tx++) {
            if (x + tx >= fxsize/2 && x + tx - fxsize/2 < xsize_in &&
                y + ty >= fysize/2 && y + ty - fysize/2 < ysize_in) {
              r += f[ty * fxsize + tx] * frame_in[((y + ty - fysize/2) * xsize_in +x + tx -fxsize/2)* bytes + b];
            } else {
              // use centre pixel when over the border
              r += f[ty * fxsize + tx] * frame_in[(y * xsize_in + x) * bytes + b];
            }
          }
        }
        // clip/saturate to uint8_t
        if (r < 0) frame_out[(y * xsize_in + x) * bytes + b] = 0;
        else if (r > UINT8_MAX) frame_out[(y * xsize_in + x) * bytes + b] = UINT8_MAX;
        else frame_out[(y * xsize_in + x) * bytes + b] = r;
      }
    }
  }
}
