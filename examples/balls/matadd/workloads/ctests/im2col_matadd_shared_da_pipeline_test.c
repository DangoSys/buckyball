#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/im2col.h>
#include <isa/int2fp.h>
#include <isa/matadd.h>
#include <isa/smatmul.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define T 16
#define IN 4
#define KS 3
#define OD (IN - KS + 1)
#define WINS (OD * OD)
#define KE (KS * KS)
#define PK ((KE + T - 1) / T * T)
#define PW ((WINS + T - 1) / T * T)
#define CL (PW * 2)

static int8_t qbank[2 * IN * IN * T] __attribute__((aligned(64)));
static int8_t wt[2 * PK * T] __attribute__((aligned(64)));
static int8_t q[2 * IN * IN];
static float dw[T] __attribute__((aligned(64)));
static int32_t bias[T] __attribute__((aligned(64)));
static float out[CL * 8] __attribute__((aligned(64)));

static void gemm(const int8_t *im, const int8_t *w, int32_t *o) {
  for (int oh = 0; oh < OD; ++oh)
    for (int ow = 0; ow < OD; ++ow)
      for (int j = 0; j < T; ++j) {
        int32_t a = 0;
        for (int k = 0; k < KE; ++k)
          a += (int32_t)im[(oh + k / KS) * IN + ow + k % KS] * w[k * T + j];
        o[(oh * OD + ow) * T + j] = a;
      }
}

int main(void) {
  int32_t expected[WINS * T], partial[WINS * T];
  for (int c = 0; c < 2; ++c) {
    for (int i = 0; i < IN * IN; ++i) {
      q[c * IN * IN + i] = (int8_t)(c * 7 + i - 8);
      for (int j = 0; j < T; ++j)
        qbank[c * IN * IN * T + i * T + j] = j ? 0 : q[c * IN * IN + i];
    }
  }
  for (int i = 0; i < 2 * PK * T; ++i)
    wt[i] = (int8_t)((i / T * 3 + i % T) % 41 - 20);
  for (int i = 0; i < T; ++i)
    dw[i] = 0.05f;
  gemm(q, wt, expected);
  gemm(q + IN * IN, wt + PK * T, partial);
  for (int i = 0; i < WINS * T; ++i)
    expected[i] += partial[i];

  bb_mem_alloc(0, IN * IN, 1);
  bb_mem_alloc(1, 2, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mem_alloc(3, 4, 1);
  bb_mem_alloc(4, 1, 1);
  bb_mem_alloc(5, 1, 1);
  bb_mem_alloc(6, 1, 1);
  bb_mem_alloc(7, 4, 1);
  bb_mvin((uintptr_t)dw, 3, 4, 1);
  bb_mvin((uintptr_t)bias, 7, 4, 1);
  bb_smatmul_bias(7, 0);

  bb_mvin((uintptr_t)qbank, 0, IN * IN, 1);
  bb_im2col(0, 1, IN, KS, 1, 0, 0, 0, 0, 0, 0, WINS);
  bb_mvin((uintptr_t)wt, 0, PK, 1);
  bb_smatmul_os(1, 0, 2, PW, T, PK, 1, 1, 0);
  bb_fence();
  bb_mvin((uintptr_t)(qbank + IN * IN * T), 0, IN * IN, 1);
  bb_im2col(0, 1, IN, KS, 1, 0, 0, 0, 0, 0, 0, WINS);
  bb_mvin((uintptr_t)(wt + PK * T), 0, PK, 1);
  bb_smatmul_os(1, 0, 4, PW, T, PK, 1, 1, 0);
  bb_fence();
  bb_matadd(2, 4, 6, CL);
  bb_fence();
  bb_int32_to_fp32(6, 3, 5, CL, 0);
  bb_mvout((uintptr_t)out, 5, CL, 1);
  bb_fence();

  for (int i = 0; i < WINS * T; ++i)
    if (fabsf(out[i] - expected[i] * dw[0]) > 1e-3f)
      return printf("shared_da pipeline mismatch %d\n", i), 1;
  return printf("im2col_matadd_shared_da pipeline PASSED\n"), 0;
}
