/*
 * shared_invalid_release_assert_test.c - expected-fail assertion test.
 *
 * This test should trip SharedMemBackend release-missing assertion because
 * core 0 releases a shared vbank that was never allocated.
 */

#include "goban.h"
#include "scu.h"

int main(void) {
  int hart = bb_get_hart_id();
  int cid = bb_get_core_id();
  if (cid == 0) {
    scu_puts(hart, "=== shared_invalid_release_assert_test starting ===\n");
    bb_mem_release(bb_shared_bank(7));
    scu_puts(hart, "ERROR: invalid release assertion did not fire\n");
    return 1;
  }
  return 0;
}
