#include "./image_processing_functions.h"

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
