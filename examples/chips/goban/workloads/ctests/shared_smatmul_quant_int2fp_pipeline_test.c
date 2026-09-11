#include "goban.h"
#include "scu.h"
#include <isa/int2fp.h>
#include <isa/quant.h>
#include <isa/smatmul.h>

enum { DIM = 16, ELEMS = DIM * DIM };
static int8_t a[BB_CORES_PER_TILE][ELEMS] __attribute__((aligned(64)));
static int8_t b[BB_CORES_PER_TILE][ELEMS] __attribute__((aligned(64)));
static int32_t bias[BB_CORES_PER_TILE][DIM] __attribute__((aligned(64)));
static uint32_t scale[BB_CORES_PER_TILE][ELEMS] __attribute__((aligned(64)));
static int8_t quant[BB_CORES_PER_TILE][ELEMS] __attribute__((aligned(64)));
static uint32_t fp[BB_CORES_PER_TILE][ELEMS] __attribute__((aligned(64)));
static volatile int core_ok[BB_CORES_PER_TILE];
static volatile int test_ok;

static uint32_t fpbits(int value) {
  if (value == 0)
    return 0;
  uint32_t sign = value < 0 ? 0x80000000u : 0;
  uint32_t mag = value < 0 ? (uint32_t)-value : (uint32_t)value;
  int exp = 0;
  for (uint32_t bits = mag; bits > 1; bits >>= 1)
    ++exp;
  return sign | (uint32_t)(exp + 127) << 23 |
         (mag - (1u << exp)) << (23 - exp);
}

int main(void) {
  int core = (int)bb_get_core_id().core;
  for (int r = 0; r < DIM; ++r)
    for (int c = 0; c < DIM; ++c) {
      a[core][r * DIM + c] = c == r ? (r % 3) + 1
                                     : (c == (r + 1) % DIM ? -1 : 0);
      b[core][r * DIM + c] = (r * 5 + c * 3 + core) % 7 - 3;
      scale[core][r * DIM + c] = 0x3f800000u;
    }
  for (int c = 0; c < DIM; ++c)
    bias[core][c] = c - 8 + core;

  int sa = bb_shared_bank(0), sb = bb_shared_bank(1);
  int sx = bb_shared_bank(2), sc = bb_shared_bank(3);
  bb_mem_alloc(sa, 1, 1);
  bb_mem_alloc(sb, 1, 1);
  bb_mem_alloc(sx, 1, 1);
  bb_mem_alloc(sc, 1, 1);
  bb_tile_barrier();
  bb_mvin((uintptr_t)a[core], sa, DIM, 1);
  bb_mvin((uintptr_t)b[core], sb, DIM, 1);
  bb_mvin((uintptr_t)bias[core], sx, 4, 1);
  bb_smatmul_bias(sx, 0);
  bb_fence();
  bb_smatmul_os(sa, sb, sc, DIM, DIM, DIM, 1, 1, 0);
  bb_fence();
  bb_mem_release(sa);
  bb_mem_release(sb);
  bb_mem_release(sx);
  bb_tile_barrier();

  int ss = bb_shared_bank(4), sq = bb_shared_bank(5), sf = bb_shared_bank(6);
  bb_mem_alloc(ss, 1, 1);
  bb_mem_alloc(sq, 1, 1);
  bb_mem_alloc(sf, 1, 1);
  bb_tile_barrier();
  bb_mvin((uintptr_t)scale[core], ss, DIM * 4, 1);
  bb_quant_i32_to_i8(sc, ss, sq, DIM * 4, 0, 0, 4, 4, 4, 0);
  bb_mvout((uintptr_t)quant[core], sq, DIM, 1);
  bb_fence();
  bb_mem_release(sq);
  bb_int32_to_fp32(sc, ss, sf, DIM * 4, 0);
  bb_mvout((uintptr_t)fp[core], sf, DIM * 4, 1);
  bb_fence();

  int ok = 1;
  for (int r = 0; r < DIM; ++r)
    for (int c = 0; c < DIM; ++c) {
      int expected = bias[core][c];
      for (int k = 0; k < DIM; ++k)
        expected += a[core][r * DIM + k] * b[core][k * DIM + c];
      if (quant[core][r * DIM + c] != expected ||
          fp[core][r * DIM + c] != fpbits(expected))
        ok = 0;
    }
  core_ok[core] = ok;
  bb_mem_release(sc);
  bb_mem_release(ss);
  bb_mem_release(sf);
  bb_tile_barrier();
  if (core == 0) {
    test_ok = 1;
    for (int i = 0; i < BB_CORES_PER_TILE; ++i)
      if (!core_ok[i])
        test_ok = 0;
    scu_puts(0, test_ok ? "shared_ball_pipeline PASSED\n"
                        : "shared_ball_pipeline FAILED\n");
  }
  bb_tile_barrier();
  return test_ok ? 0 : 1;
}
