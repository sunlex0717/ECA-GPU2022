#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

/* Kees Goossens
 * 2021-03-24 v1
 */

static uint8_t BMP_header[54] = {
  66, 77, 0, 0,  0, 0, 0, 0, 0,  0, 
  54,  0, 0, 0, 40, 0, 0, 0, 0,  0,
   0,  0, 0, 0,  0, 0, 1, 0, 24, 0,
   // rest are 0
};

// returns 0 on success
uint32_t writeBMP(char const * const file, uint8_t const * const frame, uint32_t const x_size, uint32_t const y_size, uint32_t const bitsperpixel)
{
  FILE *fd;

  // create bitmap file
  if (file == NULL || (fd = fopen(file,"wb")) == NULL) {
    printf("Error: cannot open file %s\n",file);
    return 1;
  }

  // write header
  // we always write BGR 24 bits per pixel, 3 bytes
  uint32_t const bbp = 24;
  uint64_t ulBitmapSize = (y_size * x_size * 3) + 54;
  // all values are in little-endian order (LSB first)
  BMP_header[2] = (ulBitmapSize >> 0) & 0xFF;
  BMP_header[3] = (ulBitmapSize >> 8) & 0xFF;
  BMP_header[4] = (ulBitmapSize >> 16) & 0xFF;
  BMP_header[5] = (ulBitmapSize >> 24) & 0xFF;
  BMP_header[18] = (x_size >> 0) & 0xFF;
  BMP_header[19] = (x_size >> 8) & 0xFF;
  BMP_header[22] = (y_size >> 0) & 0xFF;
  BMP_header[23] = (y_size >> 8) & 0xFF;
  BMP_header[28] = bbp; // always write 24 bits

  for (uint8_t c = 0; c < sizeof(BMP_header); c++) putc (BMP_header[c], fd);

  uint32_t const padding = x_size % 4;
  uint32_t const bytes = 3;

  printf("bmpsize %llu; %dx%d pixels of %d bits; padding %d\n",
         ulBitmapSize,x_size,y_size,bitsperpixel,padding);

  // in bitmaps the bottom line of the image is at the beginning of the file
  for (int32_t i = y_size-1; i >= 0; i--) {
    for (uint32_t j = 0; j < x_size; j++) {
      for (uint32_t b = 0; b < bytes; b++) {
        // for colour images pixels are stored in B, G, R order
        // for grey scale we write the same byte 3x
        if (bitsperpixel == 8) fputc(frame[i * x_size + j], fd);
        else fputc(frame[(i * x_size + j) * bytes +b], fd);
      }
    }
    // pad line until it's word (4 byte) aligned
    for (uint32_t j = 0; j < padding; j++) fputc(0,fd);
  }
  fclose(fd);
  return 0;
}
