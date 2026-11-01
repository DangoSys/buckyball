#include "mlir.h"
#include "vectors.h"
#include <thread.h>

extern const unsigned char thread_image[], thread_image_end[];

/* The quant_random rows of the mxint8quantize C tests through the bank
 * operation of the compiler: the packed rows must equal quantize_mxint8 of
 * quant.py (vectors.h) byte for byte. */
static uint8_t output[POLY_THREAD_CORES][QUANT_RANDOM_OUT_ROWS * 16]
    __attribute__((section(".data"), aligned(64))) = {{0}};
static unsigned done[POLY_THREAD_CORES];

static void exercise(unsigned core) {
  const uint32_t context = thread_find(THREAD_SIGNATURE, 0);
  size_t bytes = thread_image_end - thread_image;
  if (bytes > thread_query(context, THREAD_FIELD_CODE_BYTES) ||
      thread_query(context, THREAD_FIELD_SIGNATURE) != THREAD_SIGNATURE)
    __builtin_trap();
  struct mlir_mxquant_args args = {(uintptr_t)quant_random_input,
                                   (uintptr_t)output[core]};
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
  for (unsigned i = 0; i < sizeof(quant_random_expected); ++i)
    if (output[core][i] != quant_random_expected[i])
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
