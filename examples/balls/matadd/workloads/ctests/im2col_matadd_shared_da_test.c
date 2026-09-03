#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/fp2int.h>
#include <isa/im2col.h>
#include <isa/int2fp.h>
#include <isa/matadd.h>
#include <isa/smatmul.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#define T 16
#define IN 8
#define KS 3
#define OD (IN - KS + 1)
#define WINS (OD * OD)
#define KE (KS * KS)
#define PK ((KE + T - 1) / T * T)
#define PW ((WINS + T - 1) / T * T)
#define CL (PW * 2)
#define IR ((IN * IN + T - 1) / T)
static float img[2 * IN * IN] __attribute__((aligned(64)));
static int8_t wt[2 * PK * T] __attribute__((aligned(64)));
static int8_t qbank[2 * IR * T] __attribute__((aligned(64)));
static float out[CL * 8] __attribute__((aligned(64)));
static float dw[4] __attribute__((aligned(64))) = {0.05f};
static void gemm(const int8_t *im, const int8_t *w, int32_t *o) {
  for (int oh = 0; oh < OD; ++oh)
    for (int ow = 0; ow < OD; ++ow)
      for (int j = 0; j < T; ++j) {
        int32_t a = 0;
        for (int k = 0; k < KE; ++k) {
          int r = oh + k / KS, c = ow + k % KS;
          int8_t x =
              (r < 0 || c < 0 || r >= IN || c >= IN) ? 0 : im[r * IN + c];
          a += (int32_t)x * (int32_t)w[k * T + j];
        }
        o[(oh * OD + ow) * T + j] = a;
      }
}
int main(void) {
  for (int i = 0; i < IN * IN; ++i) {
    img[i] = ((i % 7) - 3) * 0.05f;
    img[IN * IN + i] = ((i % 11) - 5) * 4.0f;
  }
  for (int i = 0; i < (int)(sizeof(wt) / sizeof(wt[0])); ++i)
    wt[i] = 0;
  for (int c = 0; c < 2; ++c)
    for (int k = 0; k < KE; ++k)
      for (int j = 0; j < T; ++j)
        wt[c * PK * T + k * T + j] = (int8_t)((c * 9 + k * 3 + j) % 41 - 20);
  float maxa = 0.f;
  for (int i = 0; i < 2 * IN * IN; ++i)
    if (fabsf(img[i]) > maxa)
      maxa = fabsf(img[i]);
  float da = maxa / 127.f;
  int8_t q[2 * IN * IN];
  for (int i = 0; i < 2 * IN * IN; ++i) {
    int v = (int)(img[i] / da + (img[i] >= 0 ? 0.5f : -0.5f));
    q[i] = (int8_t)(v > 127 ? 127 : v < -128 ? -128 : v);
  }
  int32_t e0[WINS * T], e1[WINS * T], ei[WINS * T];
  gemm(q, wt, e0);
  gemm(q + IN * IN, wt + PK * T, e1);
  for (int i = 0; i < WINS * T; ++i)
    ei[i] = e0[i] + e1[i];
  bb_mvin_mmio((uintptr_t)dw, 16, 1, 4);
  bb_mem_alloc(0, 2 * IR, 4);
  bb_mem_alloc(1, 2 * IR, 1);
  bb_mem_alloc(2, 1, 2);
  bb_mem_alloc(4, 1, 2);
  bb_mem_alloc(6, 1, 2);
  bb_mem_alloc(3, 1, 2);
  bb_mvin((uintptr_t)img, 0, 2 * IR, 1);
  bb_fp2int(0, 1, 2 * IR, 0);
  bb_mvout((uintptr_t)qbank, 1, 2 * IR, 1);
  bb_fence();
  bb_mvin((uintptr_t)qbank, 0, IR, 1);
  bb_im2col(0, 1, IN, KS, 1, 0);
  bb_mvin((uintptr_t)wt, 0, PK, 1);
  bb_smatmul_os(1, 0, 2, PW, T, PK, 1, 1);
  bb_fence();
  bb_mvin((uintptr_t)(qbank + IR * T), 0, IR, 1);
  bb_im2col(0, 1, IN, KS, 1, 0);
  bb_mvin((uintptr_t)(wt + PK * T), 0, PK, 1);
  bb_smatmul_os(1, 0, 4, PW, T, PK, 1, 1);
  bb_fence();
  bb_matadd(2, 4, 6, CL);
  bb_fence();
  bb_int32_to_fp32(6, 3, 5, CL, 0);
  bb_mvout((uintptr_t)out, 5, CL, 1);
  bb_fence();
  for (int r = 0; r < WINS; ++r)
    for (int c = 0; c < T; ++c) {
      float got = out[(2 * r + c / 8) * 8 + (c % 8)];
      float exp = (float)ei[r * T + c] * da * dw[0];
      if (fabsf(got - exp) > 1e-3f)
        return printf("mismatch %d %d\n", r, c), 1;
    }
  return printf("im2col_matadd_shared_da PASSED\n"), 0;
}
