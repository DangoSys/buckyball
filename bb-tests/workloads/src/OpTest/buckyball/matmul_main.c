#include <stdint.h>
#include <stdio.h>

static void fail(void) {
  volatile uint32_t *sim_exit = (volatile uint32_t *)0x60000000;
  *sim_exit = 1;
  while (1) {
  }
}

// Called from MLIR @main after matmul completes.
// Uses the lowered memref ABI:
//   allocated, aligned, offset, sizes[0], sizes[1], strides[0], strides[1]
// Expected C[i][j] = 16.0f for all elements.
void check_result(int32_t *allocated, int32_t *aligned, int64_t offset,
                  int64_t size0, int64_t size1, int64_t stride0,
                  int64_t stride1) {
  (void)allocated;

  if (size0 != 64 || size1 != 64 || stride0 != 64 || stride1 != 1) {
    printf("FAILED: buckyball matmul unexpected memref shape "
           "(size=%dx%d stride=%dx%d)\n",
           (int)size0, (int)size1, (int)stride0, (int)stride1);
    fail();
  }

  int32_t *c = aligned + offset;
  const int32_t expected = 0x41800000;
  for (int i = 0; i < 64; ++i) {
    for (int j = 0; j < 64; ++j) {
      int32_t got = c[i * stride0 + j * stride1];
      if (got != expected) {
        printf("FAILED: buckyball matmul c[%d][%d] "
               "(expected 0x%08x, got 0x%08x)\n",
               i, j, expected, got);
        fail();
      }
    }
  }

  printf("PASSED: buckyball matmul 64x16 @ 16x64\n");
}
