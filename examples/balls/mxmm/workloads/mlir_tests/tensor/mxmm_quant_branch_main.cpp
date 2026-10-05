#include <CRunnerUtils.h>
#include <params.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
struct Results {
  StridedMemRefType<float, 2> first, last;
};
extern "C" void _mlir_ciface_quant_branch(Results *,
                                          StridedMemRefType<float, 2> *,
                                          StridedMemRefType<int8_t, 1> *,
                                          StridedMemRefType<int8_t, 1> *, bool);
int main() {
  enum { Bytes = BANK_LINES * (BANK_WIDTH / 8) };
  alignas(64) float input[16 * 64];
  for (auto &value : input)
    value = 1.0f;
  alignas(64) int8_t a[Bytes]{}, b[Bytes]{};
  memset(a, 0x38, 16 * 64);
  memset(b, 0x40, 16 * 64);
  memset(a + 16 * 64, 127, 16 * 2);
  memset(b + 16 * 64, 127, 16 * 2);
  StridedMemRefType<float, 2> x{input, input, 0, {16, 64}, {64, 1}};
  StridedMemRefType<int8_t, 1> wa{a, a, 0, {Bytes}, {1}},
      wb{b, b, 0, {Bytes}, {1}};
  for (bool condition : {false, true}) {
    Results result;
    _mlir_ciface_quant_branch(&result, &x, &wa, &wb, condition);
    for (int row = 0; row < 16; row++)
      for (int col = 0; col < 16; col++) {
        if (result.first
                    .data[result.first.offset + row * result.first.strides[0] +
                          col * result.first.strides[1]] !=
                (condition ? 64.0f : 128.0f) ||
            result.last.data[result.last.offset + row * result.last.strides[0] +
                             col * result.last.strides[1]] != 128.0f)
          return 1;
      }
    free(result.first.basePtr);
    free(result.last.basePtr);
  }
  puts("Mxmm shared packed lifetime across both branch paths PASS");
  return 0;
}
