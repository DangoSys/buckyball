#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Test bdb_counter: start/stop/read cycle counters
// Verification: if the instruction reaches TraceBall and completes without
// hanging, the test passes. The actual [CTRACE] output goes to bdb.log.

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  printf("=== bdb_counter test ===\n");

  // Test 1: basic start/stop on counter 0
  printf("Test 1: basic start/stop\n");
  bdb_counter_start(0, 0xA001);
  // Do some work to burn cycles
  volatile int x = 0;
  for (int i = 0; i < 10; i++) x += i;
  bdb_counter_stop(0);
  printf("Test 1 PASSED\n");

  // Test 2: read without stopping
  printf("Test 2: start/read/stop\n");
  bdb_counter_start(1, 0xA002);
  volatile int y = 0;
  for (int i = 0; i < 5; i++) y += i;
  bdb_counter_read(1);
  bdb_counter_stop(1);
  printf("Test 2 PASSED\n");

  // Test 3: nested counters (two levels)
  printf("Test 3: nested counters\n");
  bdb_counter_start(0, 0xB001);   // outer
    bdb_counter_start(1, 0xB002); // inner
    volatile int z = 0;
    for (int i = 0; i < 5; i++) z += i;
    bdb_counter_stop(1);          // inner done
  bdb_counter_stop(0);            // outer done
  printf("Test 3 PASSED\n");

  // Test 4: multiple independent counters
  printf("Test 4: multiple counters\n");
  bdb_counter_start(0, 0xC000);
  bdb_counter_start(1, 0xC001);
  bdb_counter_start(2, 0xC002);
  bdb_counter_start(3, 0xC003);
  bdb_counter_stop(3);
  bdb_counter_stop(2);
  bdb_counter_stop(1);
  bdb_counter_stop(0);
  printf("Test 4 PASSED\n");

  printf("bdb_counter test PASSED\n");
  return 0;

#ifdef MULTICORE
  exit(0);
#endif
}
