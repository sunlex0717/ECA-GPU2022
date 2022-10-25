#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

void greyscale(uint8_t const * const frame_in, uint32_t const xsize_in, uint32_t const ysize_in, uint32_t const bitsperpixel_in,
               uint8_t * const frame_out)
{
  uint32_t const bytes = bitsperpixel_in/8;
  // frame_in & frame_out can be the same
  // it's therefore crucial to have x & y incrementing
  for (uint32_t y = 0; y < ysize_in; y++) {
    for (uint32_t x = 0; x < xsize_in; x++) {
      uint32_t s = 0;
      for (uint32_t b = 0; b < bytes; b++) {
        s += frame_in[(y * xsize_in + x) * bytes + b];
      }
      frame_out[y * xsize_in + x] = s / bytes;
    }
  }
}
