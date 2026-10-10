#include "ant_tlink_bank.h"
#include <ant.h>
#include <params.h>
#include <tlink.h>
#include <topology.h>
_Static_assert(BB_MAIN_CORES == 1 && BB_COMPUTE_TILES >= 2 &&
                   !(BB_COMPUTE_TILES & (BB_COMPUTE_TILES - 1)),
               "TLink gate requires a power-of-two compute tile count");
extern const unsigned char ant_image[], ant_image_end[];
static uint64_t input[BB_COMPUTE_TILES][TLINK_CROSS_BYTES / 8]
    __attribute__((aligned(64)));
static uint64_t output[BB_COMPUTE_TILES][TLINK_CROSS_BYTES / 8]
    __attribute__((aligned(64)));
static unsigned ready[BB_COMPUTE_TILES], done[BB_COMPUTE_TILES];
static unsigned arrived[2];
static void barrier(unsigned phase) {
  __atomic_fetch_add(&arrived[phase], 1, __ATOMIC_ACQ_REL);
  while (__atomic_load_n(&arrived[phase], __ATOMIC_ACQUIRE) !=
         BB_COMPUTE_TILES) {
  }
}
static unsigned pattern(unsigned tile, unsigned i) {
  return (i * 17 + tile * 23 + 3) & 255;
}
static void phase(unsigned tile, unsigned operation) {
  const unsigned context = BB_ANT_CONTEXT;
  size_t bytes = ant_image_end - ant_image;
  struct ant_tlink_bank_args args = {operation, (uintptr_t)input[tile],
                                     (uintptr_t)output[tile]};
  ant_write(context, ANT_TLS, 0, &args, sizeof(args));
  uint64_t base = ant_query(context, ANT_TLS_BASE);
  struct ant_task task = {tile * 3 + operation + 1,
                          0,
                          bytes,
                          base,
                          base + ant_query(context, ANT_TLS_BYTES),
                          CORE_SIGNATURE};
  ant_acquire();
  ant_start(context, &task);
  int cancelled;
  if (ant_wait(context, task.id, &cancelled) || cancelled)
    __builtin_trap();
  ant_release();
}
static void exercise(unsigned tile) {
  uint64_t bank_bytes = tlink_query(TLINK_BANK_BYTES);
  if (tlink_query(TLINK_TILE_ID) != tile + 1 ||
      tlink_query(TLINK_TILE_COUNT) != BB_MAIN_CORES + BB_COMPUTE_TILES ||
      bank_bytes < TLINK_CROSS_BYTES ||
      tlink_query(TLINK_SHARED_BYTES) < bank_bytes * 2)
    __builtin_trap();
  size_t bytes = ant_image_end - ant_image;
  if (bytes > ant_query(BB_ANT_CONTEXT, ANT_CODE_BYTES) ||
      ant_query(BB_ANT_CONTEXT, ANT_SIGNATURE) != CORE_SIGNATURE)
    __builtin_trap();
  ant_write(BB_ANT_CONTEXT, ANT_CODE, 0, ant_image, bytes);
  for (unsigned i = 0; i < TLINK_CROSS_BYTES / 8; ++i) {
    uint64_t value = 0;
    for (unsigned j = 0; j < 8; ++j)
      value |= (uint64_t)pattern(tile, i * 8 + j) << (8 * j);
    input[tile][i] = value;
  }
  phase(tile, 0);
  barrier(0);
  unsigned next = (tile + 1) % BB_COMPUTE_TILES;
  unsigned previous = (tile + BB_COMPUTE_TILES - 1) % BB_COMPUTE_TILES;
  tlink_transfer(0, next + 1, bank_bytes - 16, 37);
  barrier(1);
  tlink_transfer(bank_bytes - 16, previous + 1, 128, 37);
  __atomic_store_n(&ready[tile], 1, __ATOMIC_RELEASE);
  while (!__atomic_load_n(&ready[next], __ATOMIC_ACQUIRE)) {
  }
  phase(tile, 1);
  for (unsigned i = 0; i < TLINK_CROSS_BYTES; ++i) {
    unsigned expected = pattern(tile, i >= 128 && i < 165 ? i - 128 : i);
    if (((unsigned char *)output[tile])[i] != expected)
      __builtin_trap();
  }
  phase(tile, 2);
  __atomic_store_n(&done[tile], 1, __ATOMIC_RELEASE);
}
int main(void) {
  uintptr_t hart;
  __asm__ volatile("csrr %0, mhartid" : "=r"(hart));
  if (hart)
    exercise(hart - 1);
  else
    for (unsigned tile = 0; tile < BB_COMPUTE_TILES; ++tile)
      while (!__atomic_load_n(&done[tile], __ATOMIC_ACQUIRE)) {
      }
  return 0;
}
