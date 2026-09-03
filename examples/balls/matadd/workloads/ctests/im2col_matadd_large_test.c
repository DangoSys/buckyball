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
#define IN 24
#define KS 3
#define ST 1
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
int main(void) {
  if (CL > 1024) {
    printf("CL=%d too big\n", CL);
    return 1;
  }
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
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, G);
  bb_mem_alloc(4, 1, G);
  bb_mem_alloc(6, 1, G);
  bb_mvin((uintptr_t)img0, 0, (IN * IN + T - 1) / T, 1);
  bb_im2col(0, 1, IN, KS, ST, 0);
  bb_mvin((uintptr_t)wt0, 0, PK, 1);
  bb_smatmul_os(1, 0, 2, PW, T, PK, 1, 1);
  bb_fence();
  bb_mvin((uintptr_t)img1, 0, (IN * IN + T - 1) / T, 1);
  bb_im2col(0, 1, IN, KS, ST, 0);
  bb_mvin((uintptr_t)wt1, 0, PK, 1);
  bb_smatmul_os(1, 0, 4, PW, T, PK, 1, 1);
  bb_fence();
  bb_matadd(2, 4, 6, CL);
  bb_fence();
  bb_mvout((uintptr_t)cout, 6, CL, 1);
  bb_fence();
  for (int r = 0; r < WINS; ++r)
    for (int c = 0; c < T; ++c) {
      int32_t got = cout[(R * r + c / 8) * 8 + (c % 8)];
      if (got != exp_[r * T + c]) {
        printf("mismatch %d %d got=%d exp=%d CL=%d\n", r, c, got,
               exp_[r * T + c], CL);
        return 1;
      }
    }
  printf("im2col_matadd_large PASSED CL=%d WINS=%d\n", CL, WINS);
  return 0;
}
