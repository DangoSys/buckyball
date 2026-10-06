#include <CRunnerUtils.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern "C" void _mlir_ciface_tensor_fma(StridedMemRefType<float, 2> *,
                                        StridedMemRefType<float, 2> *,
                                        StridedMemRefType<float, 2> *);
extern "C" void _mlir_ciface_tensor_unfused(StridedMemRefType<float, 2> *,
                                            StridedMemRefType<float, 2> *,
                                            StridedMemRefType<float, 2> *);
int main() {
  float a[8 * 64]{}, b[64 * 48]{};
  for (int r = 0; r < 8; r++) {
    uint32_t x = 0xbf800000, y = 0x3f800001;
    memcpy(a + r * 64, &x, 4);
    memcpy(a + r * 64 + 1, &y, 4);
  }
  for (int c = 0; c < 48; c++) {
    uint32_t x = 0x3f800000, y = 0x3f7ffffe;
    memcpy(b + c, &x, 4);
    memcpy(b + 48 + c, &y, 4);
  }
  StridedMemRefType<float, 2> lhs{a, a, 0, {8, 64}, {64, 1}},
      rhs{b, b, 0, {64, 48}, {48, 1}}, result;
  for (int fused = 0; fused < 2; fused++) {
    (fused ? _mlir_ciface_tensor_fma : _mlir_ciface_tensor_unfused)(&result,
                                                                    &lhs, &rhs);
    for (int i = 0; i < 384; i++) {
      uint32_t bits;
      memcpy(&bits, result.data + i, 4);
      if (bits != (fused ? 0xa8800000u : 0))
        return 1;
    }
    free(result.basePtr);
  }
  puts("Mxmm FP32 tensor M8/N48 fused/unfused PASS");
  return 0;
}
