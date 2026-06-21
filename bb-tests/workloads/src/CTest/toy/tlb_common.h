#ifndef CTEST_TOY_TLB_COMMON_H
#define CTEST_TOY_TLB_COMMON_H

#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdint.h>
#include <stdio.h>

#define TLB_PAGESIZE 4096UL
#define TLB_GB_SIZE 0x40000000UL
#define TLB_DRAM_BASE 0x80000000UL
#define TLB_ALIAS_BASE_1G 0x100000000UL
#define TLB_ALIAS_BASE_4K 0x140000000UL
#define TLB_ALIAS_BASE_REMAP 0x180000000UL

#define TLB_PTE_V 0x01UL
#define TLB_PTE_R 0x02UL
#define TLB_PTE_W 0x04UL
#define TLB_PTE_X 0x08UL
#define TLB_PTE_U 0x10UL
#define TLB_PTE_A 0x40UL
#define TLB_PTE_D 0x80UL
#define TLB_PTE_RWXAD                                                          \
  (TLB_PTE_R | TLB_PTE_W | TLB_PTE_X | TLB_PTE_A | TLB_PTE_D)

#define TLB_SATP_MODE_SV39 8UL
#define TLB_SATP_MODE_BARE 0UL
#define TLB_PMPCFG_NAPOT 0x18UL
#define TLB_PMPCFG_RWX 0x07UL

#define TLB_DIM 16

static uint64_t tlb_root[512] __attribute__((aligned(TLB_PAGESIZE)));
static uint64_t tlb_l1[512] __attribute__((aligned(TLB_PAGESIZE)));
static uint64_t tlb_l0[512] __attribute__((aligned(TLB_PAGESIZE)));

static elem_t tlb_input[TLB_DIM * TLB_DIM] __attribute__((aligned(128)));
static elem_t tlb_output[TLB_DIM * TLB_DIM] __attribute__((aligned(128)));
static elem_t tlb_input_a[TLB_DIM * TLB_DIM]
    __attribute__((aligned(TLB_PAGESIZE)));
static elem_t tlb_input_b[TLB_DIM * TLB_DIM]
    __attribute__((aligned(TLB_PAGESIZE)));

static inline void tlb_fence_rw(void) {
  asm volatile("fence rw, rw" ::: "memory");
}

static inline uint64_t tlb_leaf_pte(uintptr_t pa, uint64_t flags) {
  return (((uint64_t)pa >> 12) << 10) | TLB_PTE_V | flags;
}

static inline uint64_t tlb_table_pte(uint64_t *next) {
  return (((uint64_t)next >> 12) << 10) | TLB_PTE_V;
}

static inline unsigned tlb_vpn2(uintptr_t va) {
  return (unsigned)((va >> 30) & 0x1ff);
}

static inline unsigned tlb_vpn1(uintptr_t va) {
  return (unsigned)((va >> 21) & 0x1ff);
}

static inline unsigned tlb_vpn0(uintptr_t va) {
  return (unsigned)((va >> 12) & 0x1ff);
}

static void tlb_zero_tables(void) {
  for (int i = 0; i < 512; i++) {
    tlb_root[i] = 0;
    tlb_l1[i] = 0;
    tlb_l0[i] = 0;
  }
}

static void tlb_map_identity_1g(uintptr_t pa_base) {
  tlb_root[tlb_vpn2(pa_base)] = tlb_leaf_pte(pa_base, TLB_PTE_RWXAD);
}

static void tlb_setup_identity_4gb(void) {
  tlb_zero_tables();
  for (int i = 0; i < 4; i++) {
    tlb_map_identity_1g((uintptr_t)i * TLB_GB_SIZE);
  }
}

static int tlb_map_1g(uintptr_t va_base, uintptr_t pa_base) {
  if ((va_base & (TLB_GB_SIZE - 1)) != 0 ||
      (pa_base & (TLB_GB_SIZE - 1)) != 0) {
    printf("1GB mapping is not aligned: va=0x%lx pa=0x%lx\n",
           (unsigned long)va_base, (unsigned long)pa_base);
    return 0;
  }
  tlb_root[tlb_vpn2(va_base)] = tlb_leaf_pte(pa_base, TLB_PTE_RWXAD);
  return 1;
}

static void tlb_map_4k(uintptr_t va, uintptr_t pa) {
  tlb_root[tlb_vpn2(va)] = tlb_table_pte(tlb_l1);
  tlb_l1[tlb_vpn1(va)] = tlb_table_pte(tlb_l0);
  tlb_l0[tlb_vpn0(va)] = tlb_leaf_pte(pa, TLB_PTE_RWXAD);
}

static inline void tlb_allow_all_pmp(void) {
  uintptr_t pmpaddr = ~(uintptr_t)0;
  uintptr_t pmpcfg = TLB_PMPCFG_NAPOT | TLB_PMPCFG_RWX;
  asm volatile("csrw pmpaddr0, %0" ::"r"(pmpaddr) : "memory");
  asm volatile("csrw pmpcfg0, %0" ::"r"(pmpcfg) : "memory");
  tlb_fence_rw();
}

static void tlb_enable_sv39(void) {
  uint64_t satp_val = (TLB_SATP_MODE_SV39 << 60) | ((uint64_t)tlb_root >> 12);
  tlb_allow_all_pmp();
  tlb_fence_rw();
  asm volatile("csrw satp, %0" ::"r"(satp_val) : "memory");
  asm volatile("sfence.vma" ::: "memory");
  tlb_fence_rw();
}

static void tlb_disable_vm(void) {
  uint64_t satp_val = TLB_SATP_MODE_BARE << 60;
  tlb_fence_rw();
  asm volatile("csrw satp, %0" ::"r"(satp_val) : "memory");
  asm volatile("sfence.vma" ::: "memory");
  tlb_fence_rw();
}

static int tlb_pa_in_dram_1g(uintptr_t pa) {
  return pa >= TLB_DRAM_BASE && pa < TLB_DRAM_BASE + TLB_GB_SIZE;
}

static uintptr_t tlb_dram_alias_1g(uintptr_t pa) {
  return TLB_ALIAS_BASE_1G + (pa - TLB_DRAM_BASE);
}

static uintptr_t tlb_page_alias(uintptr_t alias_base, uintptr_t pa) {
  return alias_base + (pa & (TLB_PAGESIZE - 1));
}

static int tlb_dma_roundtrip(uintptr_t load_addr, elem_t *expected,
                             elem_t *output) {
  uint32_t bank_id = 0;
  bb_mem_alloc(bank_id, 1, 1);
  clear_u8_matrix(output, TLB_DIM, TLB_DIM);
  tlb_fence_rw();
  bb_mvin(load_addr, bank_id, TLB_DIM, 1);
  bb_mvout((uintptr_t)output, bank_id, TLB_DIM, 1);
  bb_fence();
  tlb_fence_rw();
  return compare_u8_matrices(output, expected, TLB_DIM, TLB_DIM);
}

static int tlb_finish(const char *name, int passed) {
  tlb_disable_vm();
  if (passed) {
    printf("[%s] PASSED\n", name);
    return 0;
  }
  printf("[%s] FAILED\n", name);
  return 1;
}

static void tlb_select_hart(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
}

#endif
