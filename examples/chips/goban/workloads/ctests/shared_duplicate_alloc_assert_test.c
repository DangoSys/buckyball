/*
 * shared_duplicate_alloc_assert_test.c - expected-fail assertion test.
 *
 * This test should trip SharedMemBackend duplicate allocation assertion because
 * core 0 allocates the same (hart, shared vbank, group) twice.
 */

#include "goban.h"
#include "scu.h"

int main(void) {
  int hart = bb_get_hart_id();
  int cid = bb_get_core_id();
  if (cid == 0) {
    int bank = bb_shared_bank(0);
    scu_puts(hart, "=== shared_duplicate_alloc_assert_test starting ===\n");
    bb_mem_alloc(bank, 1, 1);
    bb_mem_alloc(bank, 1, 1);
    scu_puts(hart, "ERROR: duplicate allocation assertion did not fire\n");
    return 1;
  }
  return 0;
}
