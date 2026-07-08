/*
 * shared_same_vbank_test.c - Goban shared query isolation test.
 *
 * All cores allocate the same shared vbank id. Correct behavior requires the
 * SharedMemBackend query and access paths to key mappings by (hart, vbank).
 */

#include "goban.h"
#include "scu.h"
#include <string.h>

#define ROWS 16
#define ROW_ELEMS 16
#ifndef NCORES
#define NCORES 4
#endif

static elem_t src[NCORES][ROWS * ROW_ELEMS] __attribute__((aligned(128)));
static elem_t dst[NCORES][ROWS * ROW_ELEMS] __attribute__((aligned(128)));
static volatile int core_ok[NCORES];

static void log_msg(int hart, int cid, const char *msg) {
  scu_puts(hart, "[core ");
  scu_put_u32(hart, (uint32_t)cid);
  scu_puts(hart, "] ");
  scu_puts(hart, msg);
  scu_putc(hart, '\n');
}

static void log_mismatch(int hart, int cid, int idx, elem_t got, elem_t exp) {
  scu_puts(hart, "[core ");
  scu_put_u32(hart, (uint32_t)cid);
  scu_puts(hart, "] ERROR: idx=");
  scu_put_u32(hart, (uint32_t)idx);
  scu_puts(hart, " got=");
  scu_put_u32(hart, (uint32_t)got);
  scu_puts(hart, " expected=");
  scu_put_u32(hart, (uint32_t)exp);
  scu_putc(hart, '\n');
}

int main(void) {
  int hart = bb_get_hart_id();
  int cid = bb_get_core_id();
  int bank = bb_shared_bank(0);
  elem_t pat = (elem_t)(cid + 3);

  for (int i = 0; i < ROWS * ROW_ELEMS; i++) {
    src[cid][i] = pat;
  }

  bb_mem_alloc(bank, 1, 1);
  bb_mvin((uintptr_t)src[cid], bank, ROWS, 1);
  memset(dst[cid], 0, sizeof(dst[cid]));
  bb_mvout((uintptr_t)dst[cid], bank, ROWS, 1);
  bb_fence();
  bb_mem_release(bank);

  int ok = 1;
  for (int i = 0; i < ROWS * ROW_ELEMS; i++) {
    if (dst[cid][i] != pat) {
      log_mismatch(hart, cid, i, dst[cid][i], pat);
      ok = 0;
      break;
    }
  }
  core_ok[cid] = ok;
  log_msg(hart, cid, ok ? "same-vbank shared test PASSED" : "same-vbank shared test FAILED");

  bb_barrier();

  if (cid == 0) {
    int all_ok = 1;
    for (int i = 0; i < NCORES; i++) {
      if (!core_ok[i]) {
        all_ok = 0;
      }
    }
    scu_puts(hart, "=== shared_same_vbank_test ");
    scu_puts(hart, all_ok ? "PASSED" : "FAILED");
    scu_puts(hart, " ===\n");
  }

  return core_ok[cid] ? 0 : 1;
}
