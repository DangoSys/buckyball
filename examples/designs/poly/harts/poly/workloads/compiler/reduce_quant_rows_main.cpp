#include <CRunnerUtils.h>
#include <cstdint>
#include <cstdio>

extern "C" void _mlir_ciface_quantize(StridedMemRefType<float, 2> *,
                                      StridedMemRefType<int8_t, 2> *);
alignas(64) static float input[5 * 1024];
alignas(64) static int8_t output[5 * 1056];

int main() {
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned k = 0; k < 1024; ++k)
      input[row * 1024 + k] = (k % 2 ? -3 : 2) * (1 << ((k / 32 + row) % 4));
  StridedMemRefType<float, 2> source{input, input, 0, {5, 1024}, {1024, 1}};
  StridedMemRefType<int8_t, 2> result{output, output, 0, {5, 1056}, {1056, 1}};
  _mlir_ciface_quantize(&source, &result);
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned k = 0; k < 1056; ++k) {
      int expected = k < 1024 ? (k % 2 ? -96 : 64) : 122 + (k - 1024 + row) % 4;
      if (output[row * 1056 + k] != expected) {
        printf("Quantized row mismatch at %u,%u\n", row, k);
        return 1;
      }
    }
  return 0;
}
