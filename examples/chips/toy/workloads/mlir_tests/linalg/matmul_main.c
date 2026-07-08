#include <stdint.h>
#include <stdio.h>

// Called from MLIR @main after matmul completes.
// Reads c[0][0] as i32 bit pattern to avoid fp constants in MLIR.
// c_ptr points to C[4096][1024] fp32 matrix.
// Expected c[0][0] = 1024.0f (sum of 1024 multiplications of 1.0*1.0)
//   IEEE 754: 1024.0f = 0x44800000
void check_result(int32_t *c_ptr) {
  int32_t result_bits = c_ptr[0];
  if (result_bits == 0x44800000) {
    printf("PASSED: linalg matmul 4096x1024 @ 1024x1024\n");
  } else {
    printf("FAILED: linalg matmul (expected 0x44800000, got 0x%08x)\n",
           result_bits);
  }
}
