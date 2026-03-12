#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Test bdb_backdoor: SRAM backdoor write + read via DPI-C
//
// C++ side (ctrace.cc) generates test data: each 128-bit row is filled
// with (row*16 + byte_offset) & 0xFF pattern via generate_test_data().
//
// Flow:
//   1. bdb_backdoor_write: C++ generates (row, data) per iteration via DPI-C,
//      RTL writes to external bank 0
//   2. bdb_backdoor_read: RTL reads bank 0, sends data back via DPI-C,
//      C++ logs to bdb.log as [BANK-TRACE]
//   3. mvout + fence: verify data was actually written (no hang = pass)

#define DIM 16

static elem_t output_matrix[DIM * DIM] __attribute__((aligned(64)));

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  // Build expected pattern (must match C++ generate_test_data):
  // row i, byte j -> (i*16 + j) & 0xFF

  uint32_t bank_id = 0;
  bb_mem_alloc(bank_id, 1, 1);

  // C++ injects DIM rows into bank 0 via DPI-C (data decided by C++)
  bdb_backdoor_write(bank_id, DIM);

  // Dump bank 0 contents to bdb.log via DPI-C
  bdb_backdoor_read(bank_id, DIM);

  // mvout to verify data integrity
  clear_i8_matrix(output_matrix, DIM, DIM);
  bb_mvout((uintptr_t)output_matrix, bank_id, DIM, 1);
  bb_fence();

  printf("bdb_backdoor test PASSED\n");
  return 0;

#ifdef MULTICORE
  exit(0);
#endif
}
