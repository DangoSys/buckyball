#include <CRunnerUtils.h>
#include <cstdint>
#include <cstdio>

extern "C" void _mlir_ciface_quantize(StridedMemRefType<float, 2> *,
                                      StridedMemRefType<int8_t, 2> *);
alignas(64) static float input[3 * 256];
alignas(64) static int8_t output[3 * 264 + 16];

int main() {
  for (unsigned row = 0; row < 3; ++row)
    for (unsigned k = 0; k < 256; ++k)
      input[row * 256 + k] = (k % 2 ? -3 : 2) * (1 << ((k / 32 + row) % 4));
  for (auto &byte : output)
    byte = 85;
  StridedMemRefType<float, 2> source{input, input, 0, {3, 256}, {256, 1}};
  StridedMemRefType<int8_t, 2> result{output, output, 0, {3, 264}, {264, 1}};
  _mlir_ciface_quantize(&source, &result);
  for (unsigned row = 0; row < 3; ++row)
    for (unsigned k = 0; k < 264; ++k) {
      int expected = k < 256 ? (k % 2 ? -96 : 64) : 122 + (k - 256 + row) % 4;
      if (output[row * 264 + k] != expected) {
        printf("Quantized short row mismatch at %u,%u\n", row, k);
        return 1;
      }
    }
  for (unsigned i = 3 * 264; i < sizeof(output); ++i)
    if (output[i] != 85)
      return 1;
  return 0;
}
