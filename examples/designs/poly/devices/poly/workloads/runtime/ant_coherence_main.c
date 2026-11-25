#include "ant_coherence.h"
#include <bb_harts.h>
#include <thread.h>
_Static_assert(
    BB_NUM_HARTS == POLY_THREAD_HARTS + 1 && POLY_THREAD_HARTS >= 2 &&
        !(POLY_THREAD_HARTS & (POLY_THREAD_HARTS - 1)),
    "coherence gate requires a power-of-two thread-bearing hart count");
extern const unsigned char ant_image[], ant_image_end[];
// Separate 64-byte lines/sets for data and synchronization; avoid main's
// top-of-page stack sets.
enum {
  INPUT = 8,
  OUTPUT = INPUT + POLY_THREAD_HARTS,
  READY = OUTPUT + POLY_THREAD_HARTS,
  DONE = READY + POLY_THREAD_HARTS,
  ROUND = DONE + POLY_THREAD_HARTS
};
static volatile uint64_t lines[ROUND + 1][8]
    __attribute__((section(".data"), aligned(4096))) = {0};
static uint64_t pattern(unsigned round, unsigned hart, unsigned word) {
  return ((uint64_t)round << 48) | ((uint64_t)(hart + 1) << 32) | (word + 17);
}
static void task(unsigned hart, unsigned phase, unsigned id) {
  const unsigned context = thread_find(THREAD_SIGNATURE, 0);
  struct ant_coherence_args args = {phase, (uintptr_t)lines[INPUT + hart],
                                    (uintptr_t)lines[OUTPUT + hart]};
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
static void controller(unsigned hart) {
  size_t bytes = ant_image_end - ant_image;
  const unsigned context = thread_find(THREAD_SIGNATURE, 0);
  if (bytes > thread_query(context, THREAD_FIELD_CODE_BYTES) ||
      thread_query(context, THREAD_FIELD_SIGNATURE) != THREAD_SIGNATURE)
    __builtin_trap();
  thread_tss_write(0, ant_image, bytes);
  thread_load(context, THREAD_CODE, 0, 0, (bytes + 7) & ~7u);
  task(hart, 0, 1);
  __atomic_store_n(&lines[READY + hart][0], 1, __ATOMIC_RELEASE);
  for (unsigned round = 1; round <= 2; ++round) {
    while (__atomic_load_n(&lines[ROUND][0], __ATOMIC_ACQUIRE) != round) {
    }
    task(hart, 1, round + 1);
    __atomic_store_n(&lines[DONE + hart][0], round, __ATOMIC_RELEASE);
  }
  task(hart, 2, 4);
  __atomic_store_n(&lines[DONE + hart][0], 3, __ATOMIC_RELEASE);
}
int main(void) {
  uintptr_t hartid;
  __asm__ volatile("csrr %0, mhartid" : "=r"(hartid));
  if (hartid)
    controller(hartid - 1);
  else {
    for (unsigned hart = 0; hart < POLY_THREAD_HARTS; ++hart)
      while (!__atomic_load_n(&lines[READY + hart][0], __ATOMIC_ACQUIRE)) {
      }
    for (unsigned round = 1; round <= 2; ++round) {
      for (unsigned hart = 0; hart < POLY_THREAD_HARTS; ++hart)
        for (unsigned word = 0; word < 2; ++word) {
          lines[INPUT + hart][word] = pattern(round, hart, word);
          lines[OUTPUT + hart][word] = ~pattern(round, hart, word);
        }
      __atomic_store_n(&lines[ROUND][0], round, __ATOMIC_RELEASE);
      for (unsigned hart = 0; hart < POLY_THREAD_HARTS; ++hart)
        while (__atomic_load_n(&lines[DONE + hart][0], __ATOMIC_ACQUIRE) <
               round) {
        }
      for (unsigned hart = 0; hart < POLY_THREAD_HARTS; ++hart)
        for (unsigned word = 0; word < 2; ++word)
          if (lines[OUTPUT + hart][word] != pattern(round, hart, word))
            __builtin_trap();
    }
    for (unsigned hart = 0; hart < POLY_THREAD_HARTS; ++hart)
      while (__atomic_load_n(&lines[DONE + hart][0], __ATOMIC_ACQUIRE) != 3) {
      }
  }
  return 0;
}
