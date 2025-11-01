// #include "utils/mem.h"
#include "utils/debug.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#define MEM_SIZE 100000

static uint8_t mem[MEM_SIZE] = {0};

// Load image from am-kernels (Makefile -> ./image.bin)
static long load_image(char const *img_file) {
  if (img_file == NULL) {
    printf("No image is given. Use the default build-in image.");
    return 4096; // built-in image size
  }
  FILE *fp = fopen(img_file, "rb");
  Assert(fp, "Can not open '%s'", img_file);

  // fseek: set file position pointer related to fp to a specified position
  // fseek(fp, 0, SEEK_END) positions file pointer to end of file, offset 0 bytes
  fseek(fp, 0, SEEK_END);
  // ftell: returns file size
  static long img_size = ftell(fp);

  printf("The image is %s, size = %ld\n", img_file, img_size);

  // fseek(fp, 0, SEEK_SET) positions file pointer to beginning of file, offset 0 bytes
  fseek(fp, 0, SEEK_SET);
  // Read img_size from fp to mem
  int ret = fread(mem, img_size, 1, fp);
  assert(ret == 1);

  fclose(fp);
  return img_size;
}
