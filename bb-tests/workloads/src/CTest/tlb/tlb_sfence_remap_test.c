#include "tlb_common.h"

int main(void) {
  tlb_select_hart();
  const char *name = "TLB Sfence Remap";
  printf("[%s] remap one VA after sfence.vma\n", name);

  uintptr_t input_a_pa = (uintptr_t)tlb_input_a;
  uintptr_t input_b_pa = (uintptr_t)tlb_input_b;
  uintptr_t alias_va = tlb_page_alias(TLB_ALIAS_BASE_REMAP, input_a_pa);

  init_u8_random_matrix(tlb_input_a, TLB_DIM, TLB_DIM, 201);
  init_u8_random_matrix(tlb_input_b, TLB_DIM, TLB_DIM, 202);

  tlb_setup_identity_4gb();
  tlb_map_4k(alias_va, input_a_pa);
  tlb_enable_sv39();

  int passed = tlb_dma_roundtrip(alias_va, tlb_input_a, tlb_output);
  if (!passed) {
    return tlb_finish(name, 0);
  }

  tlb_map_4k(alias_va, input_b_pa);
  asm volatile("sfence.vma" ::: "memory");

  passed = tlb_dma_roundtrip(alias_va, tlb_input_b, tlb_output);
  if (!passed) {
    printf("[%s] stale translation survived remap\n", name);
  }
  return tlb_finish(name, passed);
}
