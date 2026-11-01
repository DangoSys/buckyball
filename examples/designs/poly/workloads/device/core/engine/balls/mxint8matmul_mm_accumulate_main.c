#include "task.h"
#include "vectors.h"
#include <thread.h>

extern const unsigned char thread_image[], thread_image_end[];

/* Runs the mxint8matmul mm_accumulate thread program on the ffn thread of every
 * core; each core's C window must equal the block-ordered FP32 product of
 * vectors.h bit for bit. */
static uint32_t c[POLY_THREAD_CORES][MM_ACCUMULATE_C_ROWS * 4]
    __attribute__((section(".data"), aligned(64))) = {{0}};
static unsigned done[POLY_THREAD_CORES];

static void exercise(unsigned core) {
  const uint32_t context = thread_find(THREAD_SIGNATURE, 0);
  size_t bytes = thread_image_end - thread_image;
  if (bytes > thread_query(context, THREAD_FIELD_CODE_BYTES) ||
      thread_query(context, THREAD_FIELD_SIGNATURE) != THREAD_SIGNATURE)
    __builtin_trap();
  for (unsigned i = 0; i < MM_ACCUMULATE_C_ROWS * 4; ++i)
    c[core][i] = mm_accumulate_c[i];
  struct mxint8matmul_task args = {(uintptr_t)mm_accumulate_a,
                                   (uintptr_t)mm_accumulate_b,
                                   (uintptr_t)c[core]};
  thread_tss_write(0, thread_image, bytes);
  thread_load(context, THREAD_CODE, 0, 0, (bytes + 7) & ~7u);
  thread_tss_write(0, &args, sizeof(args));
  thread_load(context, THREAD_TLS, 0, 0, (sizeof(args) + 7) & ~7u);
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
  for (unsigned i = 0; i < MM_ACCUMULATE_M * MM_ACCUMULATE_N; ++i)
    if (c[core][MM_ACCUMULATE_C_BASE * 4 + i] != mm_accumulate_expected[i])
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
