#include "../../../../models/toy/alexnet/execute/conv.c"
#include "buckyball.h"

static float input[6] __attribute__((aligned(64))) = {0.125f,  0.375f, -0.125f,
                                                      -0.375f, 64.f,   -64.f};
static int8_t weights[6] __attribute__((aligned(64))) = {1, 2, 3, 4, 5, 6};
static float bias[1], output[1];

int main(void) {
  AlexnetF32_4 in = {input, input, 0, {1, 6, 1, 1}, {6, 1, 1, 1}};
  AlexnetI8_2 w = {weights, weights, 0, {6, 1}, {1, 1}};
  AlexnetF32_1 b = {bias, bias, 0, {1}, {1}};
  AlexnetF32_4 out = {output, output, 0, {1, 1, 1, 1}, {1, 1, 1, 1}};
  _mlir_ciface_gemmini_conv(&in, &w, &b, &out, 1, 1, 1, 0, 0.25f, 0.5f);
  // RNE codes are {0, 2, 0, -2, 127, -128}; accumulation is -137.
  if (output[0] != -17.125f) {
    printf("AlexNet Gemmini rounding FAILED: %f\n", output[0]);
    return 1;
  }
  printf("AlexNet Gemmini rounding PASSED\n");
  return 0;
}
