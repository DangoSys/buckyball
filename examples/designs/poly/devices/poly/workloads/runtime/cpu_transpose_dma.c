#include "cpu_transpose_dma.h"
#include <bb_harts.h>
#include <stdint.h>
#include <thread.h>
_Static_assert(BB_NUM_HARTS >= 2, "The visibility test needs two harts");
extern const unsigned char transpose_ant[], transpose_ant_end[];
static const uint32_t row7[32] = {
    0x40e00000, 0x40a7f9ef, 0x407bedce, 0x403ceb88, 0x400dab88, 0x3fd4799f,
    0x3f9f5576, 0x3f6ef790, 0x3f333333, 0x3f06618b, 0x3ec98b0a, 0x3e9722d2,
    0x3e62ac0d, 0x3e29fae5, 0x3dfeef24, 0x3dbf2c74, 0x3d8f5c29, 0x3d57027a,
    0x3d213c08, 0x3cf1d150, 0x3cb55671, 0x3c87fbeb, 0x3c4bf283, 0x3c18f05c,
    0x3be56042, 0x3bac01fa, 0x3b80fcd3, 0x3b417440, 0x3b1111f4, 0x3ad99312,
    0x3aa32869, 0x3a74b3c8};
static const struct {
  uint32_t prefix[4], row[32], suffix[68];
} expected __attribute__((section(".bbsim.reference"), used)) = {
    .prefix = {[0 ... 3] = 0xa5a5a5a5},
    .row = {0x40e00000, 0x40a7f9ef, 0x407bedce, 0x403ceb88, 0x400dab88,
            0x3fd4799f, 0x3f9f5576, 0x3f6ef790, 0x3f333333, 0x3f06618b,
            0x3ec98b0a, 0x3e9722d2, 0x3e62ac0d, 0x3e29fae5, 0x3dfeef24,
            0x3dbf2c74, 0x3d8f5c29, 0x3d57027a, 0x3d213c08, 0x3cf1d150,
            0x3cb55671, 0x3c87fbeb, 0x3c4bf283, 0x3c18f05c, 0x3be56042,
            0x3bac01fa, 0x3b80fcd3, 0x3b417440, 0x3b1111f4, 0x3ad99312,
            0x3aa32869, 0x3a74b3c8},
    .suffix = {[0 ... 67] = 0xa5a5a5a5}};
static const struct {
  uint64_t actual;
  const void *reference;
  uint64_t bytes;
} check __attribute__((section(".bbsim.check_output"), used)) = {
    0, &expected, sizeof(expected)};
static volatile uint32_t matrix[4096] __attribute__((aligned(4096)));
static volatile uint32_t transposed[32] __attribute__((aligned(64)));
static volatile uint32_t output[40] __attribute__((aligned(64)));
static uint64_t stage __attribute__((aligned(64)));
static void task(unsigned ctx, unsigned phase, uintptr_t src, uintptr_t dst) {
  struct transpose_dma_args args = {phase, src, dst};
  thread_tss_write(0, &args, sizeof(args));
  thread_load(ctx, THREAD_TLS, 0, 0, sizeof(args));
  uint64_t base = thread_query(ctx, THREAD_FIELD_TLS_BASE);
  size_t size = transpose_ant_end - transpose_ant;
  uint64_t limit = base + thread_query(ctx, THREAD_FIELD_TLS_BYTES);
  struct thread_task command = {phase + 1, 0,     size,
                                base,      limit, THREAD_SIGNATURE};
  thread_acquire();
  thread_start(ctx, &command);
  int cancelled;
  if (thread_wait(ctx, phase + 1, &cancelled) || cancelled)
    __builtin_trap();
  thread_release();
}
int main(void) {
  uintptr_t hart;
  asm volatile("csrr %0,mhartid" : "=r"(hart));
  if (hart == 1) {
    unsigned context = thread_find(THREAD_SIGNATURE, 0);
    size_t bytes = transpose_ant_end - transpose_ant;
    if (bytes > thread_query(context, THREAD_FIELD_CODE_BYTES))
      __builtin_trap();
    thread_tss_write(0, transpose_ant, bytes);
    thread_load(context, THREAD_CODE, 0, 0, (bytes + 7) & ~7u);
    while (__atomic_load_n(&stage, __ATOMIC_ACQUIRE) != 1) {
    }
    task(context, 0, (uintptr_t)row7, (uintptr_t)(matrix + 7));
    __atomic_store_n(&stage, 2, __ATOMIC_RELEASE);
    while (__atomic_load_n(&stage, __ATOMIC_ACQUIRE) != 3) {
    }
    task(context, 1, (uintptr_t)transposed, (uintptr_t)(output + 4));
    __atomic_store_n(&stage, 4, __ATOMIC_RELEASE);
  } else if (hart == 0) {
    for (unsigned i = 0; i < 4096; ++i)
      matrix[i] = 0xa5a5a5a5;
    for (unsigned i = 0; i < 40; ++i)
      output[i] = 0xa5a5a5a5;
    __atomic_store_n(&stage, 1, __ATOMIC_RELEASE);
    while (__atomic_load_n(&stage, __ATOMIC_ACQUIRE) != 2) {
    }
    for (unsigned i = 0; i < 32; ++i) {
      const volatile uint32_t *src = matrix + i * 128 + 7;
      volatile uint32_t *dst = transposed + i;
      asm volatile("flw fa5,0(%0)\nfsw fa5,0(%1)" ::"r"(src), "r"(dst)
                   : "fa5", "memory");
    }
    __atomic_store_n(&stage, 3, __ATOMIC_RELEASE);
    while (__atomic_load_n(&stage, __ATOMIC_ACQUIRE) != 4) {
    }
    volatile uint32_t *port = (volatile uint32_t *)(uintptr_t)0x60000000;
    *port = 0x42424f55;
    for (unsigned i = 0; i < 40; ++i)
      *port = output[i];
    for (unsigned i = 0; i < 32; ++i) {
      *port = matrix[i * 128 + 6];
      *port = matrix[i * 128 + 8];
    }
  }
  if (hart != 0)
    for (;;)
      asm volatile("wfi");
  return 0;
}
