#include "mlir.h"
#include <thread.h>

extern const unsigned char thread_image[], thread_image_end[];

// A: 16 rows of 4 FP32. B: 16 columns of 4 FP32. C: 16x16 FP32, then a guard
// row. Row r of A is {-1, 1 + 2^-23, r, 1}, column j of B {1, 1 - 2^-23, 1, j}.
struct buffers {
  uint32_t a[16 * 4];
  uint32_t b[16 * 4];
  uint32_t c[65 * 4];
};
struct all {
  buffers core[POLY_THREAD_CORES];
};
static constexpr all prepare() {
  all data{};
  for (unsigned core = 0; core < POLY_THREAD_CORES; ++core) {
    buffers &buffer = data.core[core];
    for (unsigned i = 0; i < 16; ++i) {
      buffer.a[i * 4 + 0] = __builtin_bit_cast(uint32_t, -1.0f);
      buffer.a[i * 4 + 1] = 0x3f800001;
      buffer.a[i * 4 + 2] = __builtin_bit_cast(uint32_t, (float)i);
      buffer.a[i * 4 + 3] = __builtin_bit_cast(uint32_t, 1.0f);
      buffer.b[i * 4 + 0] = __builtin_bit_cast(uint32_t, 1.0f);
      buffer.b[i * 4 + 1] = 0x3f7ffffe;
      buffer.b[i * 4 + 2] = __builtin_bit_cast(uint32_t, 1.0f);
      buffer.b[i * 4 + 3] = __builtin_bit_cast(uint32_t, (float)i);
    }
    for (unsigned i = 0; i < 65 * 4; ++i)
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
  // Fused: -1 + (1 + 2^-23)(1 - 2^-23) leaves -2^-46, which survives only
  // where r + j is 0.
  for (unsigned row = 0; row < 16; ++row)
    for (unsigned column = 0; column < 16; ++column)
      if (buffer.c[row * 16 + column] !=
          (row == 0 && column == 0
               ? 0xa8800000u
               : __builtin_bit_cast(uint32_t, (float)(row + column))))
        __builtin_trap();
  for (unsigned i = 256; i < 260; ++i)
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
