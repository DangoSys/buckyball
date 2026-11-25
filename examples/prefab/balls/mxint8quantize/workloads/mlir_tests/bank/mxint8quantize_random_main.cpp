#include <CRunnerUtils.h>
#include <cstdio>
#include <cstring>

/* Three rows of K = 544 FP32 quantized through the bank operation, every
 * block of its own power-of-two scale. Checked against the definition (as in
 * the C test quant_random): the block exponent from the largest magnitude,
 * each code within half a step of its value, the clamp at 127, the zero
 * padding. */
constexpr int rows = 3, k = 544, stride = 576;

extern "C" void _mlir_ciface_quant(StridedMemRefType<uint32_t, 2> *,
                                   StridedMemRefType<uint8_t, 2> *);
alignas(64) static float input[rows * k];
alignas(64) static uint8_t output[rows * stride];

/* 2^n, -126 <= n <= 127. */
static float power(int n) {
  uint32_t bits = (uint32_t)(n + 127) << 23;
  float value;
  memcpy(&value, &bits, 4);
  return value;
}

int main() {
  uint32_t state = 2024;
  for (int i = 0; i < rows * k; ++i) {
    state = state * 1664525u + 1013904223u;
    float unit = (float)(state >> 8) / (float)(1u << 23) - 1.0f;
    input[i] = unit * power((i / 32) * 7 % 61 - 30);
  }
  auto *data = reinterpret_cast<uint32_t *>(input);
  StridedMemRefType<uint32_t, 2> in{data, data, 0, {rows * k / 4, 4}, {4, 1}};
  StridedMemRefType<uint8_t, 2> out{
      output, output, 0, {rows * stride / 16, 16}, {16, 1}};
  _mlir_ciface_quant(&in, &out);
  for (int row = 0; row < rows; ++row) {
    const uint8_t *packed = output + row * stride;
    for (int block = 0; block < k / 32; ++block) {
      const float *x = input + row * k + block * 32;
      uint32_t largest = 0;
      for (int i = 0; i < 32; ++i) {
        uint32_t bits;
        memcpy(&bits, &x[i], 4);
        if ((bits & 0x7fffffff) > largest)
          largest = bits & 0x7fffffff;
      }
      int biased = (int)(largest >> 23);
      int e = largest == 0 ? 0 : biased == 0 ? -127 : biased - 133;
      if (e < -127)
        e = -127;
      if (packed[k + block] != (uint8_t)(e + 127)) {
        printf("quant: row %d block %d exponent %u, not %d\n", row, block,
               packed[k + block], e + 127);
        return 1;
      }
      for (int i = 0; i < 32; ++i) {
        float scaled = x[i] * power(-e);
        float target = scaled > 127.0f    ? 127.0f
                       : scaled < -127.0f ? -127.0f
                                          : scaled;
        float error = (float)(int8_t)packed[block * 32 + i] - target;
        if (error > 0.5f || error < -0.5f) {
          printf("quant: row %d value %d code %d for %f\n", row, block * 32 + i,
                 (int8_t)packed[block * 32 + i], (double)scaled);
          return 2;
        }
      }
    }
    for (int i = k + k / 32; i < stride; ++i)
      if (packed[i]) {
        printf("quant: row %d padding byte %d\n", row, i);
        return 3;
      }
  }
  printf("mxint8quantize random PASSED\n");
  return 0;
}
