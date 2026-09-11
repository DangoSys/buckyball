#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/vecmat16.h>
#include <multicore.h>

#define DIM 16

static elem_t a[DIM * DIM] __attribute__((aligned(64)));
static elem_t at[DIM * DIM] __attribute__((aligned(64)));
static elem_t b[DIM * DIM] __attribute__((aligned(64)));
static result_t out[DIM * DIM] __attribute__((aligned(64)));
static result_t expected[DIM * DIM] __attribute__((aligned(64)));
extern volatile uint32_t test_done;
extern volatile int test_result;

#ifdef __cplusplus
extern "C"
#endif
    int decode_main(core_id_t id) {
  if (id.core != 3)
    for (;;)
      asm volatile("wfi");
  for (int row = 0; row < DIM; ++row)
    for (int col = 0; col < DIM; ++col) {
      a[row * DIM + col] = (3 * row + 5 * col) % 11 - 5;
      b[row * DIM + col] = (7 * row + 2 * col) % 13 - 6;
      at[col * DIM + row] = a[row * DIM + col];
    }
  cpu_matmul(a, b, expected, DIM, DIM, DIM);
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 4);
  bb_mvin((uintptr_t)at, 0, DIM, 1);
  bb_mvin((uintptr_t)b, 1, DIM, 1);
  bb_vecmat16(0, 1, 2, DIM, 0);
  bb_mvout((uintptr_t)out, 2, DIM, 1);
  bb_fence();
  test_result = compare_u32_matrices(out, expected, DIM, DIM) ? 0 : 1;
  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);
  asm volatile("fence rw, rw" ::: "memory");
  test_done = 1;
  for (;;)
    asm volatile("wfi");
}
