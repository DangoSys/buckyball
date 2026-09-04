#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/fp2int.h>
#include <isa/int2fp.h>
#include <isa/smatmul.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define TILE_ROWS 32
#define TILE_COLS 16
#define OUTPUT_ROUNDS 2

static float activation[TILE_ROWS * TILE_COLS] __attribute__((aligned(64)));
static int8_t weight[16 * 16] __attribute__((aligned(64)));
static int32_t zero[TILE_ROWS * TILE_COLS] __attribute__((aligned(64)));
static float output[TILE_ROWS * TILE_COLS] __attribute__((aligned(64)));
static float dw[4] __attribute__((aligned(64))) = {0.25f, 0.0f, 0.0f, 0.0f};

static uint32_t f32_bits(float value) {
  union {
    float f;
    uint32_t u;
  } bits = {.f = value};
  return bits.u;
}

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  for (int i = 0; i < TILE_ROWS * TILE_COLS; ++i)
    activation[i] = 8.0f;
  for (int i = 0; i < 16 * 16; ++i)
    weight[i] = 0;
  for (int k = 0; k < 16; ++k)
    weight[k * 16] = 1;
  bb_mvin_mmio((uintptr_t)dw, 16, 1, 4);
  // Two physical groups hold each logical FP32 row in two consecutive lines.
  bb_mem_alloc(0, TILE_ROWS * 2, 2);
  bb_mem_alloc(1, TILE_ROWS, 1);
  bb_mem_alloc(2, TILE_COLS, 1);
  bb_mem_alloc(3, TILE_ROWS * OUTPUT_ROUNDS, 2);
  bb_mvin((uintptr_t)activation, 0, TILE_ROWS * 2, 1);
  bb_mvin((uintptr_t)weight, 2, TILE_COLS, 1);
  bb_mvin((uintptr_t)zero, 3, TILE_ROWS * OUTPUT_ROUNDS, 1);
  bb_fp2int(0, 1, TILE_ROWS * 2, 0);
  bb_fence();
  bb_mem_release(0);
  bb_mem_alloc(4, TILE_ROWS * OUTPUT_ROUNDS, 2);
  bb_smatmul_os(1, 2, 3, TILE_ROWS, TILE_COLS, TILE_COLS, 1, 1, 0);
  bb_int32_to_fp32(3, 5, 4, TILE_ROWS * OUTPUT_ROUNDS, 0);
  bb_mvout((uintptr_t)output, 4, TILE_ROWS * OUTPUT_ROUNDS, 1);
  bb_fence();
  int ok = 1;
  for (int row = 0; row < TILE_ROWS; ++row) {
    if (fabsf(output[row * TILE_COLS] - 32.0f) > 1e-5f) {
      printf("quant_online_gemm row=%d expected=0x42000000 actual=0x%08x\n",
             row, f32_bits(output[row * TILE_COLS]));
      ok = 0;
    }
  }
  bb_mem_release(1);
  bb_mem_release(2);
  bb_mem_release(3);
  bb_mem_release(4);
  printf("quant_online_gemm %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
