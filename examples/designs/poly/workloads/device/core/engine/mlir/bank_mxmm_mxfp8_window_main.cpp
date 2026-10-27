#include "mlir.h"
#include <thread.h>

extern const unsigned char thread_image[], thread_image_end[];

// A: one row of 64 E4M3 codes, then its two scales. B: 16 columns of 32
// codes, then 16 scales. C: one row of 16 FP32, then a guard row.
// Outside the window A holds NaN codes and a NaN scale, which the Ball must
// not read. Inside, element t is 1 + (t % 8) / 8; column j of B is
// 2 * (1 + (j % 8) / 8).
struct buffers {
  uint8_t a[5 * 16];
  uint8_t b[33 * 16];
  uint32_t c[5 * 4];
};
struct all {
  buffers core[POLY_THREAD_CORES];
};
static constexpr all prepare() {
  all data{};
  for (unsigned core = 0; core < POLY_THREAD_CORES; ++core) {
    buffers &buffer = data.core[core];
    for (unsigned t = 0; t < 64; ++t)
      buffer.a[t] = t < 32 ? 0x7f : 0x38 + t % 8;
    buffer.a[64] = 255;
    buffer.a[65] = 127;
    for (unsigned i = 0; i < 16 * 32; ++i)
      buffer.b[i] = 0x40 + (i / 32) % 8;
    for (unsigned i = 16 * 32; i < 16 * 33; ++i)
      buffer.b[i] = 127;
    for (unsigned i = 0; i < 5 * 4; ++i)
      buffer.c[i] = 0x5a5a5a5a;
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
  struct mlir_mxmm_args args = {(uintptr_t)buffer.a, (uintptr_t)buffer.b,
                                (uintptr_t)buffer.c};
  run(core, &args, sizeof(args));
  // The window sums to 46, so column j is 46 * 2 * (8 + j % 8) / 8.
  for (unsigned column = 0; column < 16; ++column)
    if (__builtin_bit_cast(float, buffer.c[column]) != 11.5f * (8 + column % 8))
      __builtin_trap();
  for (unsigned i = 16; i < 20; ++i)
    if (buffer.c[i] != 0x5a5a5a5a)
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
