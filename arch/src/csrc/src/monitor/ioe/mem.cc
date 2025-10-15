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

  fseek(fp, 0,
        SEEK_END); // fseek:把与fp有关的文件位置指针放到一个指定位置 //
                   // fseek(fp, 0, SEEK_END)文件指针定位到文件末尾，偏移0个字节
  static long img_size = ftell(fp); // ftell:返回文件大小

  printf("The image is %s, size = %ld\n", img_file, img_size);

  fseek(fp, 0,
        SEEK_SET); // fseek(fp, 0, SEEK_SET)文件指针定位到文件末尾，偏移0个字节
  int ret = fread(mem, img_size, 1, fp); // 从fp向mem读img_size大小
  assert(ret == 1);

  fclose(fp);
  return img_size;
}
