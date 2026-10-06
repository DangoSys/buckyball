#include <CRunnerUtils.h>
#include <params.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
struct Results {
  StridedMemRefType<int8_t, 1> a, b;
};
extern "C" void _mlir_ciface_quant_input(Results *,
                                         StridedMemRefType<float, 2> *,
                                         StridedMemRefType<float, 2> *);
int main() {
  alignas(64) float a[16 * 64], b[16 * 64];
  for (auto &value : a)
    value = 1.0f;
  for (auto &value : b)
    value = 2.0f;
  StridedMemRefType<float, 2> x{a, a, 0, {16, 64}, {64, 1}},
      y{b, b, 0, {16, 64}, {64, 1}};
  Results result;
  _mlir_ciface_quant_input(&result, &x, &y);
  StridedMemRefType<int8_t, 1> *buffers[] = {&result.a, &result.b};
  for (int which = 0; which < 2; which++) {
    auto &buffer = *buffers[which];
    for (int i = 0; i < 16 * 64; i++)
      if (uint8_t(buffer.data[buffer.offset + i * buffer.strides[0]]) != 0x78)
        return 1;
    for (int i = 0; i < 16 * 2; i++)
      if (uint8_t(
              buffer.data[buffer.offset + (16 * 64 + i) * buffer.strides[0]]) !=
          119 + which)
        return 2;
    free(buffer.basePtr);
  }
  puts("Mxmm distinct activation SSA inputs PASS");
  return 0;
}
