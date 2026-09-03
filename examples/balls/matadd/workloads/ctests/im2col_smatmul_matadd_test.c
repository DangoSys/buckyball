#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/im2col.h>
#include <isa/matadd.h>
#include <isa/smatmul.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define T 16
#define G 2
#define R 2
#define IN 33
#define KS 7
#define ST 2
#define OD (((IN - KS) / ST) + 1)
#define WINS (OD * OD)
#define KE (KS * KS)
#define PK ((KE + T - 1) / T * T)
#define PW ((WINS + T - 1) / T * T)
#define CL (PW * R)
static int8_t img0[IN * IN] __attribute__((aligned(64)));
static int8_t img1[IN * IN] __attribute__((aligned(64)));
static int8_t wt0[PK * T] __attribute__((aligned(64)));
static int8_t wt1[PK * T] __attribute__((aligned(64)));
static int32_t c0[CL * 8] __attribute__((aligned(64)));
static int32_t c1[CL * 8] __attribute__((aligned(64)));
static int32_t cout[CL * 8] __attribute__((aligned(64)));
static int32_t exp_[WINS * T];
static int8_t at(const int8_t *img, int r, int c) {
  return (r < 0 || c < 0 || r >= IN || c >= IN) ? 0 : img[r * IN + c];
}
static void gemm_cpu(const int8_t *img, const int8_t *wt, int32_t *o) {
  for (int oh = 0; oh < OD; ++oh)
    for (int ow = 0; ow < OD; ++ow)
      for (int j = 0; j < T; ++j) {
        int32_t acc = 0;
        for (int kr = 0; kr < KS; ++kr)
          for (int kc = 0; kc < KS; ++kc)
            acc += (int32_t)at(img, oh * ST + kr, ow * ST + kc) *
                   (int32_t)wt[(kr * KS + kc) * T + j];
        o[(oh * OD + ow) * T + j] = acc;
      }
}
static void run_cin(const int8_t *img, const int8_t *wt, int32_t *c,
                    uint32_t cb) {
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(cb, 1, G);
  bb_mvin((uintptr_t)img, 0, (IN * IN + T - 1) / T, 1);
  bb_im2col(0, 1, IN, KS, ST, 0);
  bb_mvin((uintptr_t)wt, 0, PK, 1);
  bb_smatmul_os(1, 0, cb, PW, T, PK, 1, 1);
  bb_mvout((uintptr_t)c, cb, CL, 1);
  bb_fence();
  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(cb);
}
int main(void) {
  for (int i = 0; i < IN * IN; ++i) {
    img0[i] = (int8_t)((i * 13 + 7) % 61 - 30);
    img1[i] = (int8_t)((i * 17 + 3) % 53 - 26);
  }
  memset(wt0, 0, sizeof(wt0));
  memset(wt1, 0, sizeof(wt1));
  for (int k = 0; k < KE; ++k)
    for (int j = 0; j < T; ++j) {
      wt0[k * T + j] = (int8_t)((k * 11 + j * 5) % 47 - 23);
      wt1[k * T + j] = (int8_t)((k * 19 + j * 9) % 43 - 21);
    }
  int32_t e0[WINS * T], e1[WINS * T];
  gemm_cpu(img0, wt0, e0);
  gemm_cpu(img1, wt1, e1);
  for (int i = 0; i < WINS * T; ++i)
    exp_[i] = e0[i] + e1[i];
  run_cin(img0, wt0, c0, 2);
  run_cin(img1, wt1, c1, 3);
  bb_mem_alloc(2, 1, G);
  bb_mem_alloc(3, 1, G);
  bb_mem_alloc(4, 1, G);
  bb_mvin((uintptr_t)c0, 2, CL, 1);
  bb_mvin((uintptr_t)c1, 3, CL, 1);
  bb_matadd(2, 3, 4, CL);
  bb_mvout((uintptr_t)cout, 4, CL, 1);
  bb_fence();
  for (int r = 0; r < WINS; ++r)
    for (int c = 0; c < T; ++c) {
      int32_t got = cout[(R * r + c / 8) * 8 + (c % 8)];
      if (got != exp_[r * T + c]) {
        printf("mismatch %d %d got=%d exp=%d\n", r, c, got, exp_[r * T + c]);
        return 1;
      }
    }
  printf("im2col_smatmul_matadd PASSED\n");
  return 0;
}
