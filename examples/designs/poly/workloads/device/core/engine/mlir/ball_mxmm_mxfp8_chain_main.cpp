#include "mlir.h"
#include <thread.h>

extern const unsigned char thread_image[], thread_image_end[];

// A: 16 rows of 32 E4M3 codes, then 16 scales. B: 16 columns of 32 codes,
// then 16 scales. C: 16x16 FP32 from row 1, the rest a guard pattern.
// Row r of A is 0.5 * (1 + (r % 8) / 8) (code 0x38 + r % 8, scale 2^-1);
// column j of B is 4 * (1 + (j % 8) / 8) (code 0x40 + j % 8, scale 2^1).
struct buffers {
  uint8_t a[33 * 16];
  uint8_t b[33 * 16];
  uint32_t c[65 * 4];
  uint32_t between[65 * 4];
};
struct all {
  buffers core[POLY_THREAD_CORES];
};
static constexpr all prepare() {
  all data{};
  for (unsigned core = 0; core < POLY_THREAD_CORES; ++core) {
    buffers &buffer = data.core[core];
    for (unsigned i = 0; i < 16 * 32; ++i) {
      buffer.a[i] = 0x38 + (i / 32) % 8;
      buffer.b[i] = 0x40 + (i / 32) % 8;
    }
    for (unsigned i = 16 * 32; i < 16 * 33; ++i) {
      buffer.a[i] = 126;
      buffer.b[i] = 128;
    }
    for (unsigned i = 0; i < 65 * 4; ++i)
      buffer.c[i] = 0x5a5a5a5a;
    for (unsigned i = 0; i < 65 * 4; ++i)
      buffer.between[i] = 0x5a5a5a5a;
  }
  return data;
}
alignas(64) static all data = prepare();
static unsigned done[POLY_THREAD_CORES];

static void run(unsigned core, const void *args, size_t size) {
  const uint32_t context = thread_find(THREAD_SIGNATURE, 0);
  size_t bytes = thread_image_end - thread_image;
  if (bytes > thread_query(context, THREAD_FIELD_CODE_BYTES) ||
      thread_query(context, THREAD_FIELD_SIGNATURE) != THREAD_SIGNATURE)
    __builtin_trap();
  thread_tss_write(0, thread_image, bytes);
  thread_load(context, THREAD_CODE, 0, 0, (bytes + 7) & ~7u);
  thread_tss_write(0, args, size);
  thread_load(context, THREAD_TLS, 0, 0, (size + 7) & ~7u);
  uint64_t base = thread_query(context, THREAD_FIELD_TLS_BASE),
           capacity = thread_query(context, THREAD_FIELD_TLS_BYTES);
  struct thread_task task = {core + 1,        0, bytes, base, base + capacity,
                             THREAD_SIGNATURE};
  thread_acquire();
  thread_start(context, &task);
  int cancelled;
  if (thread_wait(context, task.id, &cancelled) || cancelled)
    __builtin_trap();
  thread_release();
}

static void exercise(unsigned core) {
  buffers &buffer = data.core[core];
  struct mlir_mxmm_chain_args args = {(uintptr_t)buffer.a, (uintptr_t)buffer.b,
                                      (uintptr_t)buffer.c,
                                      (uintptr_t)buffer.between};
  run(core, &args, sizeof(args));
  // The first step does not write C.
  for (unsigned i = 0; i < 65 * 4; ++i)
    if (buffer.between[i] != 0x5a5a5a5a)
      __builtin_trap();
  for (unsigned i = 0; i < 4; ++i)
    if (buffer.c[i] != 0x5a5a5a5a)
      __builtin_trap();
  // Each step adds (8 + r % 8) * (8 + j % 8); the chain has two.
  for (unsigned row = 0; row < 16; ++row)
    for (unsigned column = 0; column < 16; ++column)
      if (__builtin_bit_cast(float, buffer.c[4 + row * 16 + column]) !=
          2.0f * (8 + row % 8) * (8 + column % 8))
        __builtin_trap();
  __atomic_store_n(&done[core], 1, __ATOMIC_RELEASE);
}

int main(void) {
  uintptr_t hart;
  __asm__ volatile("csrr %0, mhartid" : "=r"(hart));
  if (hart == WORKLOAD_REPORT_HART) {
    for (unsigned core = 0; core < POLY_THREAD_CORES; ++core)
      while (!__atomic_load_n(&done[core], __ATOMIC_ACQUIRE)) {
      }
  } else {
    exercise(hart - 1);
  }
  return 0;
}
