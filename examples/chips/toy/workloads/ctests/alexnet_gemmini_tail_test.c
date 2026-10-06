#include "../../../../models/toy/alexnet/execute/conv.c"
#include "buckyball.h"

static float input[3 * 5 * 5] __attribute__((aligned(64)));
static int8_t weights[27 * 5] __attribute__((aligned(64)));
static float bias[5], output[5 * 3 * 3];

int main(void) {
  for (int i = 0; i < 75; ++i)
    input[i] = (float)(i % 9 - 4) / 4;
  for (int i = 0; i < 135; ++i)
    weights[i] = i % 5 - 2;
  for (int i = 0; i < 5; ++i)
    bias[i] = (float)i / 2;
  AlexnetF32_4 in = {input, input, 0, {1, 3, 5, 5}, {75, 25, 5, 1}};
  AlexnetI8_2 w = {weights, weights, 0, {27, 5}, {5, 1}};
  AlexnetF32_1 b = {bias, bias, 0, {5}, {1}};
  AlexnetF32_4 out = {output, output, 0, {1, 5, 3, 3}, {45, 9, 3, 1}};
  _mlir_ciface_gemmini_conv(&in, &w, &b, &out, 3, 3, 1, 0, 0.25f, 0.125f);
  for (int n = 0; n < 5; ++n)
    for (int y = 0; y < 3; ++y)
      for (int x = 0; x < 3; ++x) {
        int sum = 0;
        for (int ky = 0; ky < 3; ++ky)
          for (int kx = 0; kx < 3; ++kx)
            for (int c = 0; c < 3; ++c) {
              int code = (c * 25 + (y + ky) * 5 + x + kx) % 9 - 4;
              sum += code * weights[((ky * 3 + kx) * 3 + c) * 5 + n];
            }
        float expected = (float)sum * 0.03125f + bias[n];
        if (output[n * 9 + y * 3 + x] != expected) {
          printf("AlexNet Gemmini tail FAILED n=%d y=%d x=%d got32=%ld "
                 "expected32=%ld\n",
                 n, y, x, (long)(output[n * 9 + y * 3 + x] * 32),
                 (long)(expected * 32));
          return 1;
        }
      }
  printf("AlexNet Gemmini tail PASSED\n");
  return 0;
}
