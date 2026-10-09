#include "ant_coherence.h"
#include <ant.h>
#include <params.h>
#include <topology.h>
_Static_assert(BB_MAIN_CORES == 1 && BB_COMPUTE_TILES >= 2 &&
                   !(BB_COMPUTE_TILES & (BB_COMPUTE_TILES - 1)),
               "coherence gate requires a power-of-two compute tile count");
extern const unsigned char ant_image[], ant_image_end[];
// Separate 64-byte lines/sets for data and synchronization; avoid main's
// top-of-page stack sets.
enum {
  INPUT = 8,
  OUTPUT = INPUT + BB_COMPUTE_TILES,
  READY = OUTPUT + BB_COMPUTE_TILES,
  DONE = READY + BB_COMPUTE_TILES,
  ROUND = DONE + BB_COMPUTE_TILES
};
static volatile uint64_t lines[ROUND + 1][8]
    __attribute__((section(".data"), aligned(4096))) = {0};
static uint64_t pattern(unsigned round, unsigned tile, unsigned word) {
  return ((uint64_t)round << 48) | ((uint64_t)(tile + 1) << 32) | (word + 17);
}
static void task(unsigned tile, unsigned phase, unsigned id) {
  const unsigned context = BB_ANT_CONTEXT;
  struct ant_coherence_args args = {phase, (uintptr_t)lines[INPUT + tile],
                                    (uintptr_t)lines[OUTPUT + tile]};
  ant_write(context, ANT_TLS, 0, &args, sizeof(args));
  uint64_t base = ant_query(context, ANT_TLS_BASE);
  struct ant_task command = {id,
                             0,
                             ant_image_end - ant_image,
                             base,
                             base + ant_query(context, ANT_TLS_BYTES),
                             CORE_SIGNATURE};
  ant_acquire();
  ant_start(context, &command);
  int cancelled;
  if (ant_wait(context, id, &cancelled) || cancelled)
    __builtin_trap();
  ant_release();
}
static void controller(unsigned tile) {
  size_t bytes = ant_image_end - ant_image;
  if (bytes > ant_query(BB_ANT_CONTEXT, ANT_CODE_BYTES) ||
      ant_query(BB_ANT_CONTEXT, ANT_SIGNATURE) != CORE_SIGNATURE)
    __builtin_trap();
  ant_write(BB_ANT_CONTEXT, ANT_CODE, 0, ant_image, bytes);
  task(tile, 0, 1);
  __atomic_store_n(&lines[READY + tile][0], 1, __ATOMIC_RELEASE);
  for (unsigned round = 1; round <= 2; ++round) {
    while (__atomic_load_n(&lines[ROUND][0], __ATOMIC_ACQUIRE) != round) {
    }
    task(tile, 1, round + 1);
    __atomic_store_n(&lines[DONE + tile][0], round, __ATOMIC_RELEASE);
  }
  task(tile, 2, 4);
  __atomic_store_n(&lines[DONE + tile][0], 3, __ATOMIC_RELEASE);
}
int main(void) {
  uintptr_t hart;
  __asm__ volatile("csrr %0, mhartid" : "=r"(hart));
  if (hart)
    controller(hart - 1);
  else {
    for (unsigned tile = 0; tile < BB_COMPUTE_TILES; ++tile)
      while (!__atomic_load_n(&lines[READY + tile][0], __ATOMIC_ACQUIRE)) {
      }
    for (unsigned round = 1; round <= 2; ++round) {
      for (unsigned tile = 0; tile < BB_COMPUTE_TILES; ++tile)
        for (unsigned word = 0; word < 2; ++word) {
          lines[INPUT + tile][word] = pattern(round, tile, word);
          lines[OUTPUT + tile][word] = ~pattern(round, tile, word);
        }
      __atomic_store_n(&lines[ROUND][0], round, __ATOMIC_RELEASE);
      for (unsigned tile = 0; tile < BB_COMPUTE_TILES; ++tile)
        while (__atomic_load_n(&lines[DONE + tile][0], __ATOMIC_ACQUIRE) <
               round) {
        }
      for (unsigned tile = 0; tile < BB_COMPUTE_TILES; ++tile)
        for (unsigned word = 0; word < 2; ++word)
          if (lines[OUTPUT + tile][word] != pattern(round, tile, word))
            __builtin_trap();
    }
    for (unsigned tile = 0; tile < BB_COMPUTE_TILES; ++tile)
      while (__atomic_load_n(&lines[DONE + tile][0], __ATOMIC_ACQUIRE) != 3) {
      }
  }
  return 0;
}
