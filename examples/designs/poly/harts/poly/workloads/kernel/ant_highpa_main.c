#include "ant_highpa.h"
#include <thread.h>
extern const unsigned char ant_image[], ant_image_end[];
static unsigned done[POLY_THREAD_HARTS];
static void exercise(unsigned hart) {
  const unsigned context = thread_find(THREAD_SIGNATURE, 0);
  uintptr_t high = 0x180200000ULL + hart * 0x20000ULL;
  volatile uint64_t *input = (void *)high, *output = (void *)(high + 0x1000);
  volatile uint64_t *low = (void *)(high & 0xffffffffULL);
  volatile uint64_t *sentinel = (void *)((high + 0x1000) & 0xffffffffULL);
  const uint64_t marker = 0xdeadface12345678ULL;
  uint64_t first = 0x1234567800000000ULL | ((uint64_t)hart << 16);
  uint64_t second = 0xfedcba9800000000ULL | ((uint64_t)hart << 16);
  for (unsigned i = 0; i < 4; ++i) {
    low[i] = second + i;
    sentinel[i] = marker;
    input[i] = first + i;
    output[i] = 0;
  }
  for (unsigned i = 0; i < 4; ++i)
    if (input[i] != first + i || low[i] != second + i || sentinel[i] != marker)
      __builtin_trap();
  size_t bytes = ant_image_end - ant_image;
  if (bytes > thread_query(context, THREAD_FIELD_CODE_BYTES) ||
      thread_query(context, THREAD_FIELD_SIGNATURE) != THREAD_SIGNATURE)
    __builtin_trap();
  thread_tss_write(0, ant_image, bytes);
  thread_load(context, THREAD_CODE, 0, 0, (bytes + 7) & ~7u);
  uint64_t base = thread_query(context, THREAD_FIELD_TLS_BASE);
  for (unsigned phase = 0; phase < 2; ++phase) {
    struct ant_highpa_args args = {phase ? (uintptr_t)low : high,
                                   high + 0x1000};
    thread_tss_write(0, &args, sizeof(args));
    thread_load(context, THREAD_TLS, 0, 0, (sizeof(args) + 7) & ~7u);
    struct thread_task task = {
        hart * 2 + phase + 1,
        0,
        bytes,
        base,
        base + thread_query(context, THREAD_FIELD_TLS_BYTES),
        THREAD_SIGNATURE};
    thread_acquire();
    thread_start(context, &task);
    int cancelled;
    if (thread_wait(context, task.id, &cancelled) || cancelled)
      __builtin_trap();
    thread_release();
    for (unsigned i = 0; i < 4; ++i)
      if (output[i] != (phase ? second : first) + i || sentinel[i] != marker ||
          input[i] != first + i || low[i] != second + i)
        __builtin_trap();
  }
  __atomic_store_n(&done[hart], 1, __ATOMIC_RELEASE);
}
int main(void) {
  uintptr_t hartid;
  __asm__ volatile("csrr %0, mhartid" : "=r"(hartid));
  if (hartid == WORKLOAD_REPORT_HART) {
    for (unsigned hart = 0; hart < POLY_THREAD_HARTS; ++hart)
      while (!__atomic_load_n(&done[hart], __ATOMIC_ACQUIRE)) {
      }
  } else {
    exercise(hartid - 1);
  }
  return 0;
}
