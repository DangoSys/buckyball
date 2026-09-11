#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/transpose.h>
#include <isa/vecmat16.h>
#include <multicore.h>
#include <stdint.h>

enum { DIM = 16 };
static int8_t a[BB_CORES_PER_TILE][DIM * DIM] __attribute__((aligned(64)));
static int8_t b[BB_CORES_PER_TILE][DIM * DIM] __attribute__((aligned(64)));
static int32_t actual[BB_CORES_PER_TILE][DIM * DIM]
    __attribute__((aligned(64)));
static int32_t expected[BB_CORES_PER_TILE][DIM * DIM]
    __attribute__((aligned(64)));

#ifdef __cplusplus
extern "C"
#endif
int decode_core(core_id_t id) {
  int8_t *lhs = a[id.core], *rhs = b[id.core];
  for (int row = 0; row < DIM; ++row)
    for (int col = 0; col < DIM; ++col) {
      lhs[row * DIM + col] = (row * 3 + col * 5) % 11 - 5;
      rhs[row * DIM + col] = (row * 7 + col * 2) % 13 - 6;
    }
  cpu_matmul(lhs, rhs, expected[id.core], DIM, DIM, DIM);
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mem_alloc(3, 1, 4);
  bb_mvin((uintptr_t)lhs, 0, DIM, 1);
  bb_mvin((uintptr_t)rhs, 2, DIM, 1);
  bb_transpose(0, 1, DIM, 8);
  bb_vecmat16(1, 2, 3, DIM, 0);
  bb_mvout((uintptr_t)actual[id.core], 3, DIM, 1);
  bb_fence();
  int ok = compare_i32_matrices(actual[id.core], expected[id.core], DIM, DIM);
  for (int bank = 0; bank < 4; ++bank)
    bb_mem_release(bank);
  return !ok;
}
