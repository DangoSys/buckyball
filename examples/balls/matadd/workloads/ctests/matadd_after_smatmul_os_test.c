#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/matadd.h>
#include <isa/smatmul.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define T 16
#define G 2
#define R 2
#define M 208
#define K 64
#define CL (M * R)

static int8_t a0[M * K] __attribute__((aligned(64)));
static int8_t a1[M * K] __attribute__((aligned(64)));
static int8_t b0[K * T] __attribute__((aligned(64)));
static int8_t b1[K * T] __attribute__((aligned(64)));
static int8_t pa[M * K] __attribute__((aligned(64)));
static int8_t pb[K * T] __attribute__((aligned(64)));
static int32_t c0[CL * 8] __attribute__((aligned(64)));
static int32_t c1[CL * 8] __attribute__((aligned(64)));
static int32_t cout[CL * 8] __attribute__((aligned(64)));
static int32_t exp_[M * T];

static void pack_a(const int8_t *s, int8_t *d) {
  for (int r = 0; r < M; ++r)
    for (int c = 0; c < K; ++c)
      d[((r / T * (K / T) + c / T) * T + r % T) * T + c % T] = s[r * K + c];
}

static void gemm_cpu(const int8_t *a, const int8_t *b, int32_t *o) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < T; ++j) {
      int32_t acc = 0;
      for (int k = 0; k < K; ++k)
        acc += (int32_t)a[i * K + k] * (int32_t)b[k * T + j];
      o[i * T + j] = acc;
    }
}

static void gemm_os(const int8_t *a, const int8_t *b, int32_t *c, uint32_t cb) {
  pack_a(a, pa);
  memcpy(pb, b, (size_t)K * T);
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(cb, 1, G);
  bb_mvin((uintptr_t)pa, 0, M * (K / T), 1);
  bb_mvin((uintptr_t)pb, 1, K, 1);
  bb_smatmul_os(0, 1, cb, M, T, K, 1, 1, 0);
  bb_mvout((uintptr_t)c, cb, CL, 1);
  bb_fence();
  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(cb);
}

int main(void) {
  for (int i = 0; i < M * K; ++i) {
    a0[i] = (int8_t)((i * 13 + 7) % 61 - 30);
    a1[i] = (int8_t)((i * 17 + 3) % 53 - 26);
  }
  for (int i = 0; i < K * T; ++i) {
    b0[i] = (int8_t)((i * 11 + 5) % 47 - 23);
    b1[i] = (int8_t)((i * 19 + 9) % 43 - 21);
  }
  int32_t e0[M * T], e1[M * T];
  gemm_cpu(a0, b0, e0);
  gemm_cpu(a1, b1, e1);
  for (int i = 0; i < M * T; ++i)
    exp_[i] = e0[i] + e1[i];
  gemm_os(a0, b0, c0, 2);
  gemm_os(a1, b1, c1, 3);
  bb_mem_alloc(2, 1, G);
  bb_mem_alloc(3, 1, G);
  bb_mem_alloc(4, 1, G);
  bb_mvin((uintptr_t)c0, 2, CL, 1);
  bb_mvin((uintptr_t)c1, 3, CL, 1);
  bb_matadd(2, 3, 4, CL);
  bb_mvout((uintptr_t)cout, 4, CL, 1);
  bb_fence();
  bb_mem_release(2);
  bb_mem_release(3);
  bb_mem_release(4);
  for (int r = 0; r < M; ++r)
    for (int c = 0; c < T; ++c) {
      int32_t got = cout[(R * r + c / 8) * 8 + (c % 8)];
      if (got != exp_[r * T + c]) {
        printf("mismatch %d %d got=%d exp=%d\n", r, c, got, exp_[r * T + c]);
        return 1;
      }
    }
  printf("matadd_after_smatmul_os PASSED\n");
  return 0;
}
