#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void fail(void) {
#ifdef BAREMETAL
  *(volatile uint32_t *)0x60000000 = 1;
  while (1) {
  }
#else
  exit(1);
#endif
}

static const int8_t expected[256] = {
    46,   -19,  -22,  11,   2,    61,   -24,  -37, 22,   2,    76,   -29,  -52,
    37,   2,    91,   63,   15,   -86,  107,  -35, 78,   15,   -101, 126,  -52,
    83,   15,   -116, 127,  -72,  88,   -47,  17,  -66,  50,   11,   -62,  22,
    -81,  65,   14,   -72,  28,   -96,  80,   14,  -82,  -102, 6,    -9,   32,
    32,   -122, 11,   -13,  42,   37,   -128, 16,  -23,  54,   42,   -128, 12,
    -89,  44,   -27,  53,   17,   -114, 59,   -42, 58,   17,   -128, 74,   -57,
    63,   23,   47,   -88,  110,  -16,  -95,  62,  -118, 127,  -15,  -100, 77,
    -128, 127,  -10,  -105, 92,   97,   -25,  39,  64,   -87,  122,  -40,  44,
    86,   -109, 127,  -55,  49,   116,  -119, 127, 48,   36,   -55,  127,  -43,
    48,   36,   -65,  127,  -59,  53,   36,   -75, 127,  -84,  58,   -79,  21,
    -31,  -16,  29,   -94,  26,   -36,  -21,  43,  -99,  31,   -41,  -26,  63,
    -104, -58,  -38,  28,   -112, 87,   -68,  -43, 43,   -128, 109,  -98,  -48,
    58,   -128, 119,  -128, 13,   -106, 94,   -56, 95,   18,   -126, 124,  -71,
    100,  13,   -128, 127,  -86,  105,  8,    47,  -88,  110,  -16,  -95,  62,
    -118, 127,  -15,  -100, 77,   -128, 127,  -10, -105, 92,   38,   12,   -29,
    88,   -12,  48,   12,   -39,  103,  -16,  61,  20,   -49,  118,  -26,  76,
    -52,  62,   -66,  64,   0,    -72,  82,   -86, 78,   0,    -82,  102,  -106,
    88,   0,    -92,  -128, 53,   -36,  -48,  43,  -128, 63,   -36,  -51,  59,
    -128, 73,   -36,  -56,  84,   -128, -58,  -38, 28,   -112, 87,   -68,  -43,
    43,   -128, 109,  -98,  -48,  58,   -128, 119, -128,
};

extern "C" void check_result(int8_t *allocated, int8_t *aligned, int64_t offset,
                             int64_t n, int64_t height, int64_t width,
                             int64_t channels, int64_t n_stride,
                             int64_t height_stride, int64_t width_stride,
                             int64_t channel_stride) {
  (void)allocated;
  if (n != 1 || height != 4 || width != 4 || channels != 16 ||
      n_stride != 256 || height_stride != 64 || width_stride != 16 ||
      channel_stride != 1)
    fail();
  int8_t *output = aligned + offset;
  int mismatch_count = 0;
  int first_y = -1, first_x = -1, first_c = -1;
  int first_expected = 0, first_actual = 0;
  for (int y = 0; y < 4; ++y)
    for (int x = 0; x < 4; ++x)
      for (int c = 0; c < 16; ++c) {
        int index = (y * 4 + x) * 16 + c;
        int actual = output[y * height_stride + x * width_stride + c];
        if (actual != expected[index]) {
          if (mismatch_count == 0) {
            first_y = y;
            first_x = x;
            first_c = c;
            first_expected = expected[index];
            first_actual = actual;
          }
          ++mismatch_count;
        }
      }
  for (int tile_y = 0; tile_y < 2; ++tile_y) {
    for (int tile_x = 0; tile_x < 2; ++tile_x) {
      uint32_t hash = 2166136261U;
      for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx)
          for (int c = 0; c < 16; ++c) {
            int y = 2 * tile_y + dy, x = 2 * tile_x + dx;
            hash ^= (uint8_t)output[y * height_stride + x * width_stride + c];
            hash *= 16777619U;
          }
      printf("tile[%d,%d]=%08x%s", tile_y, tile_x, hash,
             tile_x == 1 ? "\n" : " ");
    }
  }
  if (mismatch_count != 0) {
    printf("FAILED: mega_conv2d_s2_depthwise_s2 mismatches=%d first y=%d x=%d "
           "c=%d exp=%d got=%d\n",
           mismatch_count, first_y, first_x, first_c, first_expected,
           first_actual);
    fail();
  }
  printf("PASSED: mega_conv2d_s2_depthwise_s2 resident chain\n");
}
