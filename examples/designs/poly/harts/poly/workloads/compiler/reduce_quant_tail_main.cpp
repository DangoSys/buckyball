#include <CRunnerUtils.h>
#include <cstdint>
#include <cstdio>

extern "C" void _mlir_ciface_quantize(StridedMemRefType<float, 2> *,
                                      StridedMemRefType<int8_t, 2> *);
alignas(64) static float input[3 * 544];
alignas(64) static int8_t output[3 * 561 + 16];

int main() {
  for (unsigned row = 0; row < 3; ++row)
    for (unsigned k = 0; k < 544; ++k)
      input[row * 544 + k] = (k % 2 ? -3 : 2) * (1 << ((k / 32 + row) % 4));
  for (auto &byte : output)
    byte = 85;
  StridedMemRefType<float, 2> source{input, input, 0, {3, 544}, {544, 1}};
  StridedMemRefType<int8_t, 2> result{output, output, 0, {3, 561}, {561, 1}};
  _mlir_ciface_quantize(&source, &result);
  for (unsigned row = 0; row < 3; ++row)
    for (unsigned k = 0; k < 561; ++k) {
      int expected = k < 544 ? (k % 2 ? -96 : 64) : 122 + (k - 544 + row) % 4;
      if (output[row * 561 + k] != expected) {
        printf("Quantized tail mismatch at %u,%u\n", row, k);
        return 1;
      }
    }
  for (unsigned i = 3 * 561; i < sizeof(output); ++i)
    if (output[i] != 85)
      return 1;
  return 0;
}
