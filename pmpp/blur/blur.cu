#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>

const unsigned char CHANNELS = 3;
const int BLUR_SIZE = 7;

__global__ void blurKernel(unsigned char* Pout, unsigned char* Pin, int w, int h) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < w && row < h) {
    int pixValR = 0;
    int pixValG = 0;
    int pixValB = 0;

    int pixels = 0;

    for (int blurRow=-BLUR_SIZE; blurRow < BLUR_SIZE+1; ++blurRow) {
      for (int blurCol=-BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol) {
        int currRow = row + blurRow;
        int currCol = col + blurCol;

        if (currRow >= 0 && currRow < h && currCol >= 0 && currCol < w) {
          int offset = currRow*w + currCol;
          int rgbOffset = offset * CHANNELS;
          pixValR += Pin[rgbOffset    ];
          pixValG += Pin[rgbOffset + 1];
          pixValB += Pin[rgbOffset + 2];
          ++pixels;
        }
      }
    }

    int offset = (row*w + col) * CHANNELS;
    Pout[offset    ] = (unsigned char)(pixValR/pixels);
    Pout[offset + 1] = (unsigned char)(pixValG/pixels);
    Pout[offset + 2] = (unsigned char)(pixValB/pixels);
  }
}

void blur(unsigned char* Pin, unsigned char* Pout, int width, int height) {
  unsigned char *Pin_d, *Pout_d;
  int sizeIn = CHANNELS * width * height * sizeof(unsigned char);
  int sizeOut = CHANNELS * width * height * sizeof(unsigned char);

  cudaMalloc((void**) &Pin_d, sizeIn);
  cudaMalloc((void**) &Pout_d, sizeOut);

  cudaMemcpy(Pin_d, Pin, sizeIn, cudaMemcpyHostToDevice);
  cudaMemcpy(Pout_d, Pout, sizeOut, cudaMemcpyHostToDevice);

  dim3 dimGrid(ceil(width / 32.0), ceil(height / 32.0), 1);
  dim3 dimBlock(32, 32, 1);

  blurKernel<<<dimGrid, dimBlock>>>(Pout_d, Pin_d, width, height);
  
  cudaMemcpy(Pout, Pout_d, sizeOut, cudaMemcpyDeviceToHost);

  cudaFree(Pin_d);
  cudaFree(Pout_d);
}

void read_jpeg_file(const char *filename, unsigned char **image_data, int *width, int *height, int *channels) {
  FILE *file = fopen(filename, "rb");
  if (!file) {
    printf("Error opening the file %s\n", filename);
    return;
  }

  // Set up the JPEG decompression object and error handler
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);

  jpeg_stdio_src(&cinfo, file);
  jpeg_read_header(&cinfo, TRUE);
  jpeg_start_decompress(&cinfo);

  *width = cinfo.output_width;
  *height = cinfo.output_height;
  *channels = cinfo.output_components;

  size_t row_stride = *width * *channels;
  *image_data = (unsigned char *)malloc(row_stride * *height);

  // Read scanlines one at a time
  while (cinfo.output_scanline < cinfo.output_height) {
    unsigned char *buffer_array[1];
    buffer_array[0] = *image_data + (cinfo.output_scanline) * row_stride;
    jpeg_read_scanlines(&cinfo, buffer_array, 1);
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  fclose(file);
}

void write_jpeg_file(const char *filename, unsigned char *image_data, int width, int height, int channels, int quality) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
      printf("Error opening the file %s for writing\n", filename);
      return;
    }

    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    jpeg_stdio_dest(&cinfo, file);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = channels;
    cinfo.in_color_space = (channels == 3) ? JCS_RGB : JCS_GRAYSCALE;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    size_t row_stride = width * channels;
    while (cinfo.next_scanline < cinfo.image_height) {
      unsigned char *buffer_array[1];
      buffer_array[0] = &image_data[cinfo.next_scanline * row_stride];
      jpeg_write_scanlines(&cinfo, buffer_array, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(file);
}

int main(int argc, char *argv[]) {
  // Check if the user provided a filename
  if (argc < 2) {
    printf("Usage: %s <filename>\n", argv[0]);
    return 1;
  }

  const char *filename = argv[1];
  unsigned char *image_data;
  int width, height, channels;

  read_jpeg_file(filename, &image_data, &width, &height, &channels);

  printf("Image read successfully!\nDimensions: %d x %d, Channels: %d\n", width, height, channels);

  unsigned char* image_out = (unsigned char *)malloc(CHANNELS * width * height);

  blur(image_data, image_out, width, height);

  write_jpeg_file("output.jpg", image_out, width, height, CHANNELS, 100);

  free(image_data);
  free(image_out);
  return 0;
}
