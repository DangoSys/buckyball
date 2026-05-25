#include <stdint.h>
#include <stdio.h>

// Called from MLIR @main after matmul completes.
// Reads c[0][0] as i32 bit pattern to avoid fp constants in MLIR.
// c_ptr points to C[64][64] fp32 matrix.
// Expected c[0][0] = 16.0f (sum of 16 multiplications of 1.0*1.0)
//   IEEE 754: 16.0f = 0x41800000
void check_result(int32_t *c_ptr) {
  int32_t result_bits = c_ptr[0];
  if (result_bits == 0x41800000) {
    printf("PASSED: buckyball matmul 64x16 @ 16x64\n");
  } else {
    printf("FAILED: buckyball matmul (expected 0x41800000, got 0x%08x)\n",
           result_bits);
  }
}
