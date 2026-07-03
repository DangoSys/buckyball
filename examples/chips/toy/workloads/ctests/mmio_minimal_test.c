// Minimal MMIO test - just test mmio_set and mvin_mmio

#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <bbhw/mmio/mmio_allocator.c>
#include <bbhw/mmio/mmio_allocator.h>

static uint8_t test_data[16] __attribute__((aligned(16))) = {
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
    0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10};

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  printf("=== Minimal MMIO Test ===\n");

  // 1. Allocate MMIO region
  mmio_allocator_t mmio_alloc;
  mmio_allocator_init(&mmio_alloc);
  uint16_t mmio_addr = mmio_allocator_alloc(&mmio_alloc, 1);
  if (mmio_addr == (uint16_t)-1) {
    printf("FAIL: MMIO allocation failed\n");
    return 1;
  }
  printf("Allocated MMIO at 0x%x\n", mmio_addr);

  // 2. Allocate a bank
  uint32_t bank_id = 0;
  bb_mem_alloc(bank_id, 1, 1);
  printf("Allocated bank %d\n", bank_id);

  // 3. Bind bank to MMIO
  printf("Calling bb_mmio_set...\n");
  bb_mmio_set(bank_id, mmio_addr, 1);
  printf("bb_mmio_set done\n");

  // 4. Load data to MMIO
  printf("Calling bb_mvin_mmio...\n");
  bb_mvin_mmio((uintptr_t)test_data, mmio_addr, 1, 16);
  printf("bb_mvin_mmio done\n");

  // 5. Fence
  printf("Calling bb_fence...\n");
  bb_fence();
  printf("bb_fence done\n");

  // 6. Cleanup
  bb_mmio_set(bank_id, 0, 0);
  bb_mem_release(bank_id);
  mmio_allocator_free(&mmio_alloc, mmio_addr, 1);

  printf("MMIO test PASSED\n");

#ifdef MULTICORE
  exit(0);
#endif

  return 0;
}
