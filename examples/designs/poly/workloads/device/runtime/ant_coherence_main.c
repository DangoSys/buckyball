#include "ant_coherence.h"
#include <bb_cores.h>
#include <thread.h>
_Static_assert(
    BB_NUM_CORES == POLY_THREAD_CORES + 1 && POLY_THREAD_CORES >= 2 &&
        !(POLY_THREAD_CORES & (POLY_THREAD_CORES - 1)),
    "coherence gate requires a power-of-two thread-bearing core count");
extern const unsigned char ant_image[], ant_image_end[];
// Separate 64-byte lines/sets for data and synchronization; avoid main's
// top-of-page stack sets.
enum {
  INPUT = 8,
  OUTPUT = INPUT + POLY_THREAD_CORES,
  READY = OUTPUT + POLY_THREAD_CORES,
  DONE = READY + POLY_THREAD_CORES,
  ROUND = DONE + POLY_THREAD_CORES
};
static volatile uint64_t lines[ROUND + 1][8]
    __attribute__((section(".data"), aligned(4096))) = {0};
static uint64_t pattern(unsigned round, unsigned core, unsigned word) {
  return ((uint64_t)round << 48) | ((uint64_t)(core + 1) << 32) | (word + 17);
}
static void task(unsigned core, unsigned phase, unsigned id) {
  const unsigned context = thread_find(THREAD_SIGNATURE, 0);
  struct ant_coherence_args args = {phase, (uintptr_t)lines[INPUT + core],
                                    (uintptr_t)lines[OUTPUT + core]};
  thread_tss_write(0, &args, sizeof(args));
  thread_load(context, THREAD_TLS, 0, 0, (sizeof(args) + 7) & ~7u);
  uint64_t base = thread_query(context, THREAD_FIELD_TLS_BASE);
  struct thread_task command = {
      id,
      0,
      ant_image_end - ant_image,
      base,
      base + thread_query(context, THREAD_FIELD_TLS_BYTES),
      THREAD_SIGNATURE};
  thread_acquire();
  thread_start(context, &command);
  int cancelled;
  if (thread_wait(context, id, &cancelled) || cancelled)
    __builtin_trap();
  thread_release();
}
static void controller(unsigned core) {
  size_t bytes = ant_image_end - ant_image;
  const unsigned context = thread_find(THREAD_SIGNATURE, 0);
  if (bytes > thread_query(context, THREAD_FIELD_CODE_BYTES) ||
      thread_query(context, THREAD_FIELD_SIGNATURE) != THREAD_SIGNATURE)
    __builtin_trap();
  thread_tss_write(0, ant_image, bytes);
  thread_load(context, THREAD_CODE, 0, 0, (bytes + 7) & ~7u);
  task(core, 0, 1);
  __atomic_store_n(&lines[READY + core][0], 1, __ATOMIC_RELEASE);
  for (unsigned round = 1; round <= 2; ++round) {
    while (__atomic_load_n(&lines[ROUND][0], __ATOMIC_ACQUIRE) != round) {
    }
    task(core, 1, round + 1);
    __atomic_store_n(&lines[DONE + core][0], round, __ATOMIC_RELEASE);
  }
  task(core, 2, 4);
  __atomic_store_n(&lines[DONE + core][0], 3, __ATOMIC_RELEASE);
}
int main(void) {
  uintptr_t hart;
  __asm__ volatile("csrr %0, mhartid" : "=r"(hart));
  if (hart)
    controller(hart - 1);
  else {
    for (unsigned core = 0; core < POLY_THREAD_CORES; ++core)
      while (!__atomic_load_n(&lines[READY + core][0], __ATOMIC_ACQUIRE)) {
      }
    for (unsigned round = 1; round <= 2; ++round) {
      for (unsigned core = 0; core < POLY_THREAD_CORES; ++core)
        for (unsigned word = 0; word < 2; ++word) {
          lines[INPUT + core][word] = pattern(round, core, word);
          lines[OUTPUT + core][word] = ~pattern(round, core, word);
        }
      __atomic_store_n(&lines[ROUND][0], round, __ATOMIC_RELEASE);
      for (unsigned core = 0; core < POLY_THREAD_CORES; ++core)
        while (__atomic_load_n(&lines[DONE + core][0], __ATOMIC_ACQUIRE) <
               round) {
        }
      for (unsigned core = 0; core < POLY_THREAD_CORES; ++core)
        for (unsigned word = 0; word < 2; ++word)
          if (lines[OUTPUT + core][word] != pattern(round, core, word))
            __builtin_trap();
    }
    for (unsigned core = 0; core < POLY_THREAD_CORES; ++core)
      while (__atomic_load_n(&lines[DONE + core][0], __ATOMIC_ACQUIRE) != 3) {
      }
  }
  return 0;
}
