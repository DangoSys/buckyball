#include "../../../../models/toy/alexnet/execute/conv.c"
#include "buckyball.h"

static float input[4 * 4 * 4] __attribute__((aligned(64)));
static int8_t weights[16 * 32] __attribute__((aligned(64)));
static float bias[32], output[32 * 3 * 3];

int main(void) {
  for (int i = 0; i < 64; ++i)
    input[i] = (float)(i % 9 - 4) / 4;
  for (int i = 0; i < 512; ++i)
    weights[i] = i % 5 - 2;
  for (int i = 0; i < 32; ++i)
    bias[i] = (float)i / 2;
  AlexnetF32_4 in = {input, input, 0, {1, 4, 4, 4}, {64, 16, 4, 1}};
  AlexnetI8_2 w = {weights, weights, 0, {16, 32}, {32, 1}};
  AlexnetF32_1 b = {bias, bias, 0, {32}, {1}};
  AlexnetF32_4 out = {output, output, 0, {1, 32, 3, 3}, {288, 9, 3, 1}};
  _mlir_ciface_gemmini_conv(&in, &w, &b, &out, 2, 2, 1, 0, 0.25f, 0.125f);
  for (int n = 0; n < 32; ++n)
    for (int y = 0; y < 3; ++y)
      for (int x = 0; x < 3; ++x) {
        int sum = 0;
        for (int ky = 0; ky < 2; ++ky)
          for (int kx = 0; kx < 2; ++kx)
            for (int c = 0; c < 4; ++c) {
              int code = (c * 16 + (y + ky) * 4 + x + kx) % 9 - 4;
              sum += code * weights[((ky * 2 + kx) * 4 + c) * 32 + n];
            }
        if (output[n * 9 + y * 3 + x] != (float)sum * 0.03125f + bias[n]) {
          printf("AlexNet Gemmini stride FAILED n=%d y=%d x=%d\n", n, y, x);
          return 1;
        }
      }
  printf("AlexNet Gemmini stride PASSED\n");
  return 0;
}
