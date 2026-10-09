#include "ant_tlink_reduce.h"
#include <ant.h>
#include <bbhw/isa/isa.h>
#include <isa/f32add.h>
#include <params.h>
#include <tlink.h>
#include <topology.h>
_Static_assert(BB_MAIN_CORES == 1 && BB_COMPUTE_TILES >= 2 &&
                   (BB_COMPUTE_TILES & (BB_COMPUTE_TILES - 1)) == 0,
               "Reduction needs main and a power-of-two compute tile count");
static float input[BB_COMPUTE_TILES][272] __attribute__((aligned(64)));
static float output[BB_COMPUTE_TILES][16] __attribute__((aligned(64)));
static unsigned ready[BB_COMPUTE_TILES], broadcast, done[BB_COMPUTE_TILES];
enum { WINDOW = 4 };
static uint64_t shared_base[BB_COMPUTE_TILES], main_base[WINDOW];
static unsigned main_ready, batch;
static float partial(unsigned tile, unsigned i) {
  return (float)(i + 1) *
         (tile % 2 ? -(float)(tile + 1) * 0.5f : (float)(tile + 1) * 1.25f);
}
static void controller(unsigned tile) {
  ant_tlink_reduce_prepare(tile, input[tile], output[tile]);
  struct ant_tlink_reduce_args args = {0, (uintptr_t)input[tile],
                                       (uintptr_t)output[tile]};
  ant_tlink_reduce_phase(tile, &args);
  shared_base[tile] = tlink_shared_export(0, BB_SHARED_BANK_BASE, 0);
  while (!__atomic_load_n(&main_ready, __ATOMIC_ACQUIRE) ||
         __atomic_load_n(&batch, __ATOMIC_ACQUIRE) != tile / WINDOW) {
  }
  tlink_transfer(shared_base[tile], 0, main_base[tile % WINDOW], 16);
  __atomic_store_n(&ready[tile], 1, __ATOMIC_RELEASE);
  while (!__atomic_load_n(&broadcast, __ATOMIC_ACQUIRE)) {
  }
  args.phase = 1;
  ant_tlink_reduce_phase(tile, &args);
  for (unsigned i = 0; i < 4; ++i) {
    float expected = 0.0f;
    for (unsigned rank = 0; rank < BB_COMPUTE_TILES; ++rank)
      expected += partial(rank, i);
    if (output[tile][i] != partial(tile, i) || output[tile][i + 4] != expected)
      __builtin_trap();
  }
  tlink_shared_release(0, BB_SHARED_BANK_BASE, 0);
  args.phase = 2;
  ant_tlink_reduce_phase(tile, &args);
  __atomic_store_n(&done[tile], 1, __ATOMIC_RELEASE);
}
static void reduce(void) {
  unsigned groups = BB_COMPUTE_TILES < WINDOW ? BB_COMPUTE_TILES : WINDOW;
  unsigned source = 32, accumulator = 33, target = 34;
  bb_mem_alloc(source, 1, groups);
  bb_mem_alloc(accumulator, 1, 1);
  bb_mem_alloc(target, 1, 1);
  for (unsigned group = 0; group < groups; ++group)
    main_base[group] = tlink_shared_export(0, source, group);
  __atomic_store_n(&main_ready, 1, __ATOMIC_RELEASE);
  for (unsigned tile = 0; tile < BB_COMPUTE_TILES; ++tile) {
    while (!__atomic_load_n(&ready[tile], __ATOMIC_ACQUIRE)) {
    }
    bb_f32add(source, accumulator, target, 1, tile % WINDOW, tile == 0);
    unsigned previous = accumulator;
    accumulator = target;
    target = previous;
    if (tile % WINDOW == WINDOW - 1) {
      bb_fence();
      __atomic_store_n(&batch, tile / WINDOW + 1, __ATOMIC_RELEASE);
    }
  }
  bb_fence();
  uint64_t result = tlink_shared_export(0, accumulator, 0);
  for (unsigned tile = 0; tile < BB_COMPUTE_TILES; ++tile)
    tlink_transfer(result, tile + 1, shared_base[tile] + 16, 16);
  __atomic_store_n(&broadcast, 1, __ATOMIC_RELEASE);
  for (unsigned tile = 0; tile < BB_COMPUTE_TILES; ++tile)
    while (!__atomic_load_n(&done[tile], __ATOMIC_ACQUIRE)) {
    }
  tlink_shared_release(0, accumulator, 0);
  for (unsigned group = 0; group < groups; ++group)
    tlink_shared_release(0, source, group);
  bb_mem_release(source);
  bb_mem_release(accumulator);
  bb_mem_release(target);
}

int main(void) {
  uintptr_t hart;
  __asm__ volatile("csrr %0, mhartid" : "=r"(hart));
  if (hart > BB_COMPUTE_TILES || tlink_query(TLINK_TILE_ID) != hart ||
      tlink_query(TLINK_TILE_COUNT) != BB_COMPUTE_TILES + 1 ||
      tlink_query(TLINK_BANK_BYTES) < 48 ||
      tlink_query(TLINK_SHARED_BYTES) <
          tlink_query(TLINK_BANK_BYTES) + BB_COMPUTE_TILES * 16)
    __builtin_trap();
  if (hart)
    controller(hart - 1);
  else
    reduce();
  return 0;
}
