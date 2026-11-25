// softmax on the dense thread alone: 2 FP32 rows of 16 from bank row 1 to
// bank row 2 of another bank, scaled by 2^3. Row 0 is one value 16 times:
// every term is exp(0) = 1, so every output is 1/16. Row 1 is 0 at lane 5 and
// -inf elsewhere: output 1 at lane 5 and 0 elsewhere. Both are exact, so the
// bits are known without an FPU.
#include <bbhw/isa/isa.h>
#include <isa/softmax.h>
#include <stdint.h>

enum { R = 2, W = 16, ROWS = SOFTMAX_ROWS(W) };
#define X ((volatile uint32_t *)HOST_TSS_BASE)
#define Y ((volatile uint32_t *)(HOST_TSS_BASE + (1 + R * ROWS) * 16))

uint64_t entry(void) {
  for (int i = 0; i < W; ++i) {
    X[4 + i] = 0x40400000u;                         // 3.0
    X[4 + ROWS * 4 + i] = i == 5 ? 0 : 0xff800000u; // -inf
  }
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin((uintptr_t)X, 1, 1 + R * ROWS, 1);
  bb_softmax(1, 2, R, W, 1, 2, 3);
  bb_mvout((uintptr_t)Y, 2, 2 + R * ROWS, 1);
  asm volatile("fence" ::: "memory");
  for (int i = 0; i < W; ++i) {
    if (Y[8 + i] != 0x3d800000u) // 1/16
      return 1 + i;
    if (Y[8 + ROWS * 4 + i] != (i == 5 ? 0x3f800000u : 0))
      return 100 + i;
  }
  bb_mem_release(1);
  bb_mem_release(2);
  return 0;
}
