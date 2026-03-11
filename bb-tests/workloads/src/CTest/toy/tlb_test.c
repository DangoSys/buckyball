#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>
#include <stdlib.h>

// ---------------------------------------------------------------------------
// Sv39 page table setup for bare-metal TLB test
// ---------------------------------------------------------------------------
// Sv39: 3-level page table, 4KB pages
// VA bits: [38:30] VPN[2], [29:21] VPN[1], [20:12] VPN[0], [11:0] offset
// We use 1GB megapages (level-2 only) for simplicity: one PTE covers 1GB.
// This creates an identity mapping (VA == PA) for the first 4GB.

#define PAGESIZE 4096
#define PTE_V 0x01 // Valid
#define PTE_R 0x02 // Read
#define PTE_W 0x04 // Write
#define PTE_X 0x08 // Execute
#define PTE_U 0x00 // Supervisor only (U=0)
#define PTE_A 0x40 // Accessed
#define PTE_D 0x80 // Dirty
#define PTE_RWXAD (PTE_R | PTE_W | PTE_X | PTE_A | PTE_D)

#define SATP_MODE_SV39 8UL
#define SATP_MODE_BARE 0UL

// Page table: one page, 512 entries (level-2 table for Sv39)
// Must be page-aligned (4KB)
static uint64_t page_table[512] __attribute__((aligned(PAGESIZE)));

static void setup_identity_mapping(void) {
  // Create identity mapping using 1GB megapages
  // PTE format: [53:10] = PPN, [7:0] = flags
  // For a 1GB megapage at level 2: PPN = physical GB index
  for (int i = 0; i < 4; i++) {
    // PPN for 1GB megapage: GB_index << 18 (since PPN has 28 bits for Sv39,
    // and 1GB = 2^30 bytes, PPN = PA >> 12, so PPN = i << 18)
    uint64_t ppn = (uint64_t)i << 18;
    page_table[i] = (ppn << 10) | PTE_V | PTE_RWXAD;
  }
  // Remaining entries are invalid (0)
  for (int i = 4; i < 512; i++) {
    page_table[i] = 0;
  }
}

static void enable_sv39(void) {
  uint64_t satp_val = (SATP_MODE_SV39 << 60) | ((uint64_t)page_table >> 12);
  asm volatile("csrw satp, %0" ::"r"(satp_val));
  // Flush TLB after changing satp
  asm volatile("sfence.vma");
}

static void disable_vm(void) {
  uint64_t satp_val = SATP_MODE_BARE << 60;
  asm volatile("csrw satp, %0" ::"r"(satp_val));
  asm volatile("sfence.vma");
}

// ---------------------------------------------------------------------------
// Test data
// ---------------------------------------------------------------------------
#define DIM 16

static elem_t input_matrix[DIM * DIM] __attribute__((aligned(128)));
static elem_t output_matrix[DIM * DIM] __attribute__((aligned(128)));

// ---------------------------------------------------------------------------
// Test 1: DMA with VM disabled (satp.mode=Bare)
// Verifies passthrough works (vm_enabled=false → vaddr=paddr)
// ---------------------------------------------------------------------------
int test_bare_mode(void) {
  printf("[TLB Test 1] DMA with VM disabled (Bare mode)...\n");

  disable_vm();

  uint32_t bank_id = 0;
  bb_mem_alloc(bank_id, 1, 1);

  init_u8_random_matrix(input_matrix, DIM, DIM, 42);
  clear_u8_matrix(output_matrix, DIM, DIM);

  bb_mvin((uintptr_t)input_matrix, bank_id, DIM, 1);
  bb_mvout((uintptr_t)output_matrix, bank_id, DIM, 1);
  bb_fence();

  if (!compare_u8_matrices(output_matrix, input_matrix, DIM, DIM)) {
    printf("[TLB Test 1] FAILED: mvin/mvout mismatch in Bare mode\n");
    return 0;
  }

  printf("[TLB Test 1] PASSED\n");
  return 1;
}

// ---------------------------------------------------------------------------
// Test 2: DMA with Sv39 identity mapping
// Verifies TLB translation works (vm_enabled=true, but VA==PA)
// ---------------------------------------------------------------------------
int test_sv39_identity(void) {
  printf("[TLB Test 2] DMA with Sv39 identity mapping...\n");

  setup_identity_mapping();
  enable_sv39();

  uint32_t bank_id = 0;
  bb_mem_alloc(bank_id, 1, 1);

  init_u8_random_matrix(input_matrix, DIM, DIM, 77);
  clear_u8_matrix(output_matrix, DIM, DIM);

  bb_mvin((uintptr_t)input_matrix, bank_id, DIM, 1);
  bb_mvout((uintptr_t)output_matrix, bank_id, DIM, 1);
  bb_fence();

  if (!compare_u8_matrices(output_matrix, input_matrix, DIM, DIM)) {
    printf("[TLB Test 2] FAILED: mvin/mvout mismatch with Sv39\n");
    disable_vm();
    return 0;
  }

  printf("[TLB Test 2] PASSED\n");
  return 1;
}

// ---------------------------------------------------------------------------
// Test 3: sfence.vma flushes Buckyball TLB
// After Sv39 is active, issue sfence.vma, then verify DMA still works
// (TLB entries were flushed, must refill from page table)
// ---------------------------------------------------------------------------
int test_sfence_flush(void) {
  printf("[TLB Test 3] sfence.vma flush + DMA refill...\n");

  // sfence.vma: flush all TLBs (CPU + Buckyball)
  asm volatile("sfence.vma");

  uint32_t bank_id = 0;
  bb_mem_alloc(bank_id, 1, 1);

  init_u8_random_matrix(input_matrix, DIM, DIM, 99);
  clear_u8_matrix(output_matrix, DIM, DIM);

  bb_mvin((uintptr_t)input_matrix, bank_id, DIM, 1);
  bb_mvout((uintptr_t)output_matrix, bank_id, DIM, 1);
  bb_fence();

  if (!compare_u8_matrices(output_matrix, input_matrix, DIM, DIM)) {
    printf("[TLB Test 3] FAILED: mvin/mvout mismatch after sfence\n");
    disable_vm();
    return 0;
  }

  printf("[TLB Test 3] PASSED\n");
  return 1;
}

// ---------------------------------------------------------------------------
// Test 4: Multiple sfence cycles
// Repeatedly: do DMA → sfence → do DMA, to stress TLB flush/refill
// ---------------------------------------------------------------------------
int test_repeated_sfence(void) {
  printf("[TLB Test 4] Repeated sfence + DMA cycles...\n");

  uint32_t bank_id = 0;

  for (int i = 0; i < 4; i++) {
    asm volatile("sfence.vma");

    bb_mem_alloc(bank_id, 1, 1);
    init_u8_random_matrix(input_matrix, DIM, DIM, 100 + i);
    clear_u8_matrix(output_matrix, DIM, DIM);

    bb_mvin((uintptr_t)input_matrix, bank_id, DIM, 1);
    bb_mvout((uintptr_t)output_matrix, bank_id, DIM, 1);
    bb_fence();

    if (!compare_u8_matrices(output_matrix, input_matrix, DIM, DIM)) {
      printf("[TLB Test 4] FAILED at iteration %d\n", i);
      disable_vm();
      return 0;
    }
  }

  printf("[TLB Test 4] PASSED\n");
  return 1;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  int all_passed = 1;

  all_passed &= test_bare_mode();
  all_passed &= test_sv39_identity();
  all_passed &= test_sfence_flush();
  all_passed &= test_repeated_sfence();

  // Restore Bare mode
  disable_vm();

  if (all_passed) {
    printf("\n=== ALL TLB TESTS PASSED ===\n");
  } else {
    printf("\n=== SOME TLB TESTS FAILED ===\n");
  }

#ifdef MULTICORE
  exit(0);
#endif
  return !all_passed;
}
