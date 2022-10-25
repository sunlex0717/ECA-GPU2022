#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

/* Kees Goossens
 * 2021-03-24 v1
 */

// makes it easier to port
#define readByte fgetc(fd)

// returns 0 on success
// set the x & size and bits per pixel, malloc the frame and put the pixels in the frame
// pixels are stored per pixel in B, G, R order in the array (greyscale only 1)
uint32_t readBMP(char const * const file, uint8_t ** frame, uint32_t * const x_size, uint32_t * const y_size, uint32_t * const bitsperpixel)
{
  *x_size = 0;
  *y_size = 0;
  FILE *fd = NULL;

  // open bitmap file
  if (file == NULL || (fd = fopen(file,"r")) == NULL) {
    printf("Error: cannot open file %s\n",file);
    return 0;
  }

  // read header
  // all values are in little-endian order (LSB first)

  // BMP signature + file size
  // fprintf(fpBMP,"%c%c%c%c%c%c%c%c%c%c", 66, 77, ucaBitmapSize[0], ucaBitmapSize[1], ucaBitmapSize[2], ucaBitmapSize[3], 0, 0, 0, 0);
  (void) readByte;
  (void) readByte;
  uint32_t bmpsize = readByte;
  bmpsize |= readByte << 8;
  bmpsize |= readByte << 16;
  bmpsize |= readByte << 24;
  (void) readByte;
  (void) readByte;
  (void) readByte;
  (void) readByte;

  // image offset, infoheader size, image width
  // fprintf(fpBMP,"%c%c%c%c%c%c%c%c%c%c", 54, 0, 0, 0, 40, 0 , 0, 0, (x_size & 0x00FF), (x_size & 0xFF00)>>8);
  // image height, number of panels, num bits per pixel
  // fprintf(fpBMP,"%c%c%c%c%c%c%c%c%c%c", 0, 0, (y_size & 0x00FF), (y_size & 0xFF00) >> 8, 0, 0, 1, 0, 24, 0);
  uint32_t offset = readByte;
  offset |= readByte << 8;
  offset |= readByte << 16;
  offset |= readByte << 24;
  for (uint32_t i=0; i < 4; i++) (void) readByte;
  *x_size = readByte;
  *x_size |= readByte << 8;
  (void) readByte;
  (void) readByte;
  *y_size = readByte;
  *y_size |= readByte << 8;
  (void) readByte;
  (void) readByte;
  (void) readByte;
  (void) readByte;
  *bitsperpixel = readByte;
  (void) readByte;

  (void) readByte;
  (void) readByte;
  (void) readByte;
  (void) readByte;
  uint32_t imgsize;
  imgsize = readByte;
  imgsize |= readByte << 8;
  imgsize |= readByte << 16;
  imgsize |= readByte << 24;

  // compression type 0, Size of image in bytes 0 because uncompressed
  // fprintf(fpBMP,"%c%c%c%c%c%c%c%c%c%c", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  // fprintf(fpBMP,"%c%c%c%c%c%c%c%c%c%c", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  // fprintf(fpBMP,"%c%c%c%c", 0, 0 ,0, 0);
  for (uint32_t i=0; i < 16; i++) (void) readByte;

  // skip remainder of header
  for (uint32_t i=0; i < offset-54; i++) (void) readByte;

  uint32_t const padding = (4 - (*x_size * *bitsperpixel/8) % 4) % 4;
  uint32_t const bytesize = (*y_size)*(*x_size) * (*bitsperpixel/8);
  printf("bmpsize %u; imgsize %u; offset %d; %dx%d pixels of %d bits; padding %d\n",
         bmpsize,imgsize,offset,*x_size,*y_size,*bitsperpixel,padding);

  *frame = (uint8_t *) malloc(bytesize);
  if (frame == NULL) {
    printf("Error: cannot malloc %u bytes\n",bytesize);
    return 0;
  }

  uint32_t const bytes = *bitsperpixel/8;
  // in bitmaps the bottom line of the image is at the beginning of the file
  for (int32_t i = *y_size-1; i >= 0; i--) {
    for (uint32_t j = 0; j < *x_size; j++) {
      for (uint32_t p=0; p < bytes; p++) {
        int32_t b = readByte;
        if (b == EOF) {
          printf("Error: file too short i=%d j=%d p=%d\n",i,j,p);
          return 0;
        }
        (*frame)[(i * *x_size + j)*bytes + p] = b;
        // for colour images pixels are stored in B, G, R order
      }
    }
    // ignore the 0-3 bytes padding for every line
    if (i != 0) for (uint32_t k = 0; k < padding; k++) (void) readByte;
  }
  if (fd) fclose(fd);
  return 1;
}