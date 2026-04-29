#ifndef GOBAN_H
#define GOBAN_H

#include <bbhw/isa/isa.h>
#include <stdint.h>

/*
 * Goban's hardware barrier is tile-local. The default Goban tile currently has
 * four RocketBB cores, while BuckyballGoban2TileConfig exposes global mhartid
 * values 0..7. Tests should use the tile-local core id for per-tile arrays and
 * bank ownership.
 */
#ifndef GOBAN_CORES_PER_TILE
#define GOBAN_CORES_PER_TILE 4
#endif

#ifndef GOBAN_MAX_TILES
#define GOBAN_MAX_TILES 2
#endif

#define GOBAN_MAX_HARTS (GOBAN_CORES_PER_TILE * GOBAN_MAX_TILES)
#define GOBAN_SHARED_BANK_BASE 32

static inline int bb_get_hart_id(void) {
  int hartid;
  asm volatile("csrr %0, mhartid" : "=r"(hartid));
  return hartid;
}

static inline int bb_get_tile_core_id(void) {
  return bb_get_hart_id() % GOBAN_CORES_PER_TILE;
}

static inline int bb_get_tile_id(void) {
  return bb_get_hart_id() / GOBAN_CORES_PER_TILE;
}

static inline int bb_get_core_id(void) { return bb_get_tile_core_id(); }

static inline int bb_shared_bank(int vbank_id) {
  return GOBAN_SHARED_BANK_BASE + vbank_id;
}

static inline void bb_cpu_fence(void) {
  asm volatile("fence rw, rw" ::: "memory");
}

static inline void bb_tile_barrier(void) {
  bb_cpu_fence();
  bb_barrier();
  bb_cpu_fence();
}

#endif // GOBAN_H
