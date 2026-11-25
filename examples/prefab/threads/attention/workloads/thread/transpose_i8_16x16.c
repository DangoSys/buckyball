// transpose on the thread alone: a 16x16 INT8 matrix in the TSS, transposed
// through the banks and checked by the thread.
#include <bbhw/isa/isa.h>
#include <isa/transpose.h>
#include <stdint.h>

enum { ROWS = 16, COLS = 16 };
#define INPUT ((volatile int8_t *)HOST_TSS_BASE)
#define OUTPUT ((volatile int8_t *)(HOST_TSS_BASE + ROWS * COLS))

uint64_t entry(void) {
  for (int i = 0; i < ROWS * COLS; ++i) {
    INPUT[i] = (int8_t)(i + 1);
    OUTPUT[i] = 0;
  }
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mvin((uintptr_t)INPUT, 0, ROWS, 1);
  bb_transpose(0, 1, ROWS, 8);
  bb_mvout((uintptr_t)OUTPUT, 1, ROWS, 1);
  asm volatile("fence" ::: "memory");
  for (int row = 0; row < ROWS; ++row)
    for (int col = 0; col < COLS; ++col)
      if (OUTPUT[col * ROWS + row] != INPUT[row * COLS + col])
        return 1 + row * COLS + col;
  bb_mem_release(0);
  bb_mem_release(1);
  return 0;
}
