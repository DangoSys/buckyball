#include <CRunnerUtils.h>
#include <params.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
struct Results {
  StridedMemRefType<float, 2> gate, up;
};
extern "C" void _mlir_ciface_quant_shared(Results *,
                                          StridedMemRefType<float, 2> *,
                                          StridedMemRefType<int8_t, 1> *,
                                          StridedMemRefType<int8_t, 1> *);
int main() {
  enum { Bytes = BANK_LINES * (BANK_WIDTH / 8) };
  alignas(64) float input[16 * 64];
  alignas(64) int8_t gate[Bytes]{}, up[Bytes]{};
  for (auto &value : input)
    value = 1.0f;
  memset(gate, 0x38, 16 * 64);
  memset(up, 0x40, 16 * 64);
  memset(gate + 16 * 64, 127, 16 * 2);
  memset(up + 16 * 64, 127, 16 * 2);
  StridedMemRefType<float, 2> x{input, input, 0, {16, 64}, {64, 1}};
  StridedMemRefType<int8_t, 1> a{gate, gate, 0, {Bytes}, {1}},
      b{up, up, 0, {Bytes}, {1}};
  Results result;
  _mlir_ciface_quant_shared(&result, &x, &a, &b);
  for (int row = 0; row < 16; row++)
    for (int col = 0; col < 16; col++) {
      if (result.gate.data[result.gate.offset + row * result.gate.strides[0] +
                           col * result.gate.strides[1]] != 64.0f ||
          result.up.data[result.up.offset + row * result.up.strides[0] +
                         col * result.up.strides[1]] != 128.0f)
        return 1;
    }
  free(result.gate.basePtr);
  free(result.up.basePtr);
  puts("Mxmm shared activation: two matrix consumers PASS");
  return 0;
}
