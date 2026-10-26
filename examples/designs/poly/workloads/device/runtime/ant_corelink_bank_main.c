#include "ant_corelink_bank.h"
#include <corelink.h>
#include <instances.h>
#include <params.h>
#include <thread.h>
_Static_assert(
    BB_CORE_NUM == BB_THREAD_CORE_COUNT + 1 && BB_THREAD_CORE_COUNT >= 2 &&
        !(BB_THREAD_CORE_COUNT & (BB_THREAD_CORE_COUNT - 1)),
    "CoreLink gate requires a power-of-two thread-bearing core count");
extern const unsigned char ant_image[], ant_image_end[];
static uint64_t input[BB_THREAD_CORE_COUNT][CORELINK_CROSS_BYTES / 8]
    __attribute__((aligned(64)));
static uint64_t output[BB_THREAD_CORE_COUNT][CORELINK_CROSS_BYTES / 8]
    __attribute__((aligned(64)));
static unsigned ready[BB_THREAD_CORE_COUNT], done[BB_THREAD_CORE_COUNT];
static unsigned arrived[2];
static void barrier(unsigned phase) {
  __atomic_fetch_add(&arrived[phase], 1, __ATOMIC_ACQ_REL);
  while (__atomic_load_n(&arrived[phase], __ATOMIC_ACQUIRE) !=
         BB_THREAD_CORE_COUNT) {
  }
}
static unsigned pattern(unsigned core, unsigned i) {
  return (i * 17 + core * 23 + 3) & 255;
}
static void phase(unsigned core, unsigned operation) {
  const unsigned context = thread_find(THREAD_SIGNATURE, 0);
  size_t bytes = ant_image_end - ant_image;
  struct ant_corelink_bank_args args = {operation, (uintptr_t)input[core],
                                        (uintptr_t)output[core]};
  thread_tss_write(0, &args, sizeof(args));
  thread_load(context, THREAD_TLS, 0, 0, (sizeof(args) + 7) & ~7u);
  uint64_t base = thread_query(context, THREAD_FIELD_TLS_BASE);
  struct thread_task task = {core * 3 + operation + 1,
                             0,
                             bytes,
                             base,
                             base +
                                 thread_query(context, THREAD_FIELD_TLS_BYTES),
                             THREAD_SIGNATURE};
  thread_acquire();
  thread_start(context, &task);
  int cancelled;
  if (thread_wait(context, task.id, &cancelled) || cancelled)
    __builtin_trap();
  thread_release();
}
static void exercise(unsigned core) {
  uint64_t bank_bytes = corelink_query(CORELINK_BANK_BYTES);
  if (corelink_query(CORELINK_CORE_ID) != core + 1 ||
      corelink_query(CORELINK_CORE_COUNT) != BB_CORE_NUM ||
      bank_bytes < CORELINK_CROSS_BYTES ||
      corelink_query(CORELINK_SHARED_BYTES) < bank_bytes * 2)
    __builtin_trap();
  size_t bytes = ant_image_end - ant_image;
  const unsigned context = thread_find(THREAD_SIGNATURE, 0);
  if (bytes > thread_query(context, THREAD_FIELD_CODE_BYTES) ||
      thread_query(context, THREAD_FIELD_SIGNATURE) != THREAD_SIGNATURE)
    __builtin_trap();
  thread_tss_write(0, ant_image, bytes);
  thread_load(context, THREAD_CODE, 0, 0, (bytes + 7) & ~7u);
  for (unsigned i = 0; i < CORELINK_CROSS_BYTES / 8; ++i) {
    uint64_t value = 0;
    for (unsigned j = 0; j < 8; ++j)
      value |= (uint64_t)pattern(core, i * 8 + j) << (8 * j);
    input[core][i] = value;
  }
  phase(core, 0);
  barrier(0);
  unsigned next = (core + 1) % BB_THREAD_CORE_COUNT;
  unsigned previous = (core + BB_THREAD_CORE_COUNT - 1) % BB_THREAD_CORE_COUNT;
  corelink_transfer(0, next + 1, bank_bytes - 16, 37);
  barrier(1);
  corelink_transfer(bank_bytes - 16, previous + 1, 128, 37);
  __atomic_store_n(&ready[core], 1, __ATOMIC_RELEASE);
  while (!__atomic_load_n(&ready[next], __ATOMIC_ACQUIRE)) {
  }
  phase(core, 1);
  for (unsigned i = 0; i < CORELINK_CROSS_BYTES; ++i) {
    unsigned expected = pattern(core, i >= 128 && i < 165 ? i - 128 : i);
    if (((unsigned char *)output[core])[i] != expected)
      __builtin_trap();
  }
  phase(core, 2);
  __atomic_store_n(&done[core], 1, __ATOMIC_RELEASE);
}
int main(void) {
  uintptr_t hart;
  __asm__ volatile("csrr %0, mhartid" : "=r"(hart));
  if (hart)
    exercise(hart - 1);
  else
    for (unsigned core = 0; core < BB_THREAD_CORE_COUNT; ++core)
      while (!__atomic_load_n(&done[core], __ATOMIC_ACQUIRE)) {
      }
  return 0;
}
