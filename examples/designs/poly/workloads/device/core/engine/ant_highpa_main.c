#include "ant_highpa.h"
#include <instances.h>
#include <params.h>
#include <thread.h>
extern const unsigned char ant_image[], ant_image_end[];
static unsigned done[BB_THREAD_CORE_COUNT];
static void exercise(unsigned core) {
  const unsigned context = thread_find(THREAD_SIGNATURE, 0);
  uintptr_t high = 0x180200000ULL + core * 0x20000ULL;
  volatile uint64_t *input = (void *)high, *output = (void *)(high + 0x1000);
  volatile uint64_t *low = (void *)(high & 0xffffffffULL);
  volatile uint64_t *sentinel = (void *)((high + 0x1000) & 0xffffffffULL);
  const uint64_t marker = 0xdeadface12345678ULL;
  uint64_t first = 0x1234567800000000ULL | ((uint64_t)core << 16);
  uint64_t second = 0xfedcba9800000000ULL | ((uint64_t)core << 16);
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
        core * 2 + phase + 1,
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
  __atomic_store_n(&done[core], 1, __ATOMIC_RELEASE);
}
int main(void) {
  uintptr_t hart;
  __asm__ volatile("csrr %0, mhartid" : "=r"(hart));
  if (hart == BB_REPORT_HART) {
    for (unsigned core = 0; core < BB_THREAD_CORE_COUNT; ++core)
      while (!__atomic_load_n(&done[core], __ATOMIC_ACQUIRE)) {
      }
  } else {
    exercise(hart - 1);
  }
  return 0;
}
