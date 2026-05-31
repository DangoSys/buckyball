/*
 * Smoke test for heterogeneous Goban topologies with 4 cores per tile.
 *
 * Every hart enters main(). Only tile-local core 0 has Buckyball in this
 * topology, so Rocket-only cores must not issue custom accelerator commands.
 */

#include "goban.h"
#include "scu.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>

#define DIM 16
#define NCORES 4
#define READY_MAGIC 1U
#define READY_TIMEOUT 0x4000000U
#ifndef NTILES
#define NTILES 64
#endif
#ifndef HIDDEN_HART_BASE
#define HIDDEN_HART_BASE 64
#endif

static elem_t src[DIM * DIM] __attribute__((aligned(128)));
static elem_t dst[DIM * DIM] __attribute__((aligned(128)));

static int expected_hart_id(int tile, int cid) {
  if (cid == 0) {
    return tile;
  }
  return HIDDEN_HART_BASE + tile * (NCORES - 1) + (cid - 1);
}

static const char *hart_role(int hart) {
  if (hart < HIDDEN_HART_BASE) {
    return "visible";
  }
  return "hidden";
}

static void mark_ready(int hart) { scu_set_ready(hart, READY_MAGIC); }

static int wait_all_harts_ready(void) {
  for (uint32_t wait = 0; wait < READY_TIMEOUT; wait++) {
    int missing = -1;
    for (int hart = 0; hart < NTILES * NCORES; hart++) {
      if (scu_get_ready(hart) != READY_MAGIC) {
        missing = hart;
        break;
      }
    }
    if (missing < 0) {
      return 1;
    }
    asm volatile("nop" ::: "memory");
  }

  for (int hart = 0; hart < NTILES * NCORES; hart++) {
    uint32_t value = scu_get_ready(hart);
    if (value != READY_MAGIC) {
      printf("[hart 0] ready timeout: hart=%d value=0x%x\n", hart, value);
      return 0;
    }
  }

  return 0;
}

static void print_online(int hart, int tile, int cid) {
  scu_puts(hart, "[hart ");
  scu_put_u32(hart, (uint32_t)hart);
  scu_puts(hart, "] role=");
  scu_puts(hart, hart_role(hart));
  scu_puts(hart, " tile=");
  scu_put_u32(hart, (uint32_t)tile);
  scu_puts(hart, " cid=");
  scu_put_u32(hart, (uint32_t)cid);
  scu_puts(hart, " online\n");
}

static void fill(elem_t *buf, elem_t value) {
  for (int i = 0; i < DIM * DIM; i++) {
    buf[i] = value;
  }
}

static int same(elem_t *a, elem_t *b) {
  for (int i = 0; i < DIM * DIM; i++) {
    if (a[i] != b[i]) {
      return 0;
    }
  }
  return 1;
}

int main(void) {
  int hart = bb_get_hart_id();
  int tile = bb_get_tile_id();
  int cid = bb_get_tile_core_id();

  if (hart < 0 || hart >= NTILES * NCORES || tile < 0 || tile >= NTILES ||
      cid < 0 || cid >= NCORES) {
    printf("[hart %d] invalid ids: tile=%d cid=%d\n", hart, tile, cid);
    return 1;
  }

  if (hart != expected_hart_id(tile, cid)) {
    printf("[hart %d] unexpected mapping: tile=%d cid=%d expected=%d\n", hart,
           tile, cid, expected_hart_id(tile, cid));
    return 1;
  }

  print_online(hart, tile, cid);

  if (hart != 0) {
    scu_puts(hart, "[hart ");
    scu_put_u32(hart, (uint32_t)hart);
    scu_puts(hart, "] role=");
    scu_puts(hart, hart_role(hart));
    scu_puts(hart, " parked\n");
    mark_ready(hart);
    while (1) {
      asm volatile("wfi");
    }
  }

  mark_ready(hart);
  if (!wait_all_harts_ready()) {
    return 1;
  }

  if (tile != 0 || cid != 0) {
    printf("[hart 0] unexpected ids: tile=%d cid=%d\n", tile, cid);
    return 1;
  }

  fill(src, 7);
  fill(dst, 0);

  bb_mem_alloc(0, 1, 1);
  bb_mvin((uintptr_t)src, 0, DIM, 1);
  bb_mvout((uintptr_t)dst, 0, DIM, 1);
  bb_fence();
  bb_mem_release(0);

  if (!same(src, dst)) {
    printf("=== goban_hetero_t4c_smoke FAILED ===\n");
    return 1;
  }

  printf("=== goban_hetero_t4c_smoke PASSED ===\n");
  return 0;
}
