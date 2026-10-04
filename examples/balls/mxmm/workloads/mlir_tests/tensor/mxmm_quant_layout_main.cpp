#include <CRunnerUtils.h>
#include <params.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
struct Results {
  StridedMemRefType<int8_t, 1> row16, row32, k32, stride;
};
extern "C" void _mlir_ciface_quant_layout(Results *,
                                          StridedMemRefType<float, 2> *);
int main() {
  alignas(64) float input[16 * 64];
  for (auto &value : input)
    value = 1.0f;
  StridedMemRefType<float, 2> x{input, input, 0, {16, 64}, {64, 1}};
  Results result;
  _mlir_ciface_quant_layout(&result, &x);
  enum { Bytes = BANK_LINES * (BANK_WIDTH / 8) };
  StridedMemRefType<int8_t, 1> *buffers[] = {&result.row16, &result.row32,
                                             &result.k32, &result.stride};
  int rows[] = {16, 32, 16, 16}, ks[] = {64, 64, 32, 32},
      strides[] = {Bytes, Bytes, Bytes, Bytes / 2};
  for (int which = 0; which < 4; which++) {
    auto &buffer = *buffers[which];
    for (int panel = 0; panel < 64 / ks[which]; panel++)
      for (int row = 0; row < rows[which]; row++) {
        int base = panel * strides[which], width = ks[which];
        for (int k = 0; k < width; k++)
          if (uint8_t(buffer.data[buffer.offset + (base + row * width + k) *
                                                      buffer.strides[0]]) !=
              (row < 16 ? 0x78 : 0))
            return 1;
        for (int k = 0; k < width / 32; k++)
          if (uint8_t(buffer.data[buffer.offset + (base + rows[which] * width +
                                                   row * (width / 32) + k) *
                                                      buffer.strides[0]]) !=
              (row < 16 ? 119 : 127))
            return 2;
      }
    free(buffer.basePtr);
  }
  puts("Mxmm distinct rows/K/stride layouts: defined payload PASS");
  return 0;
}
