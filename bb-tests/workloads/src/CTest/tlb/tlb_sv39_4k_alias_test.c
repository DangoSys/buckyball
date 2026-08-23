#include "tlb_common.h"

int main(void) {
  tlb_select_hart();
  const char *name = "TLB Sv39 4KB Alias";
  printf("[%s] 4KB VA alias maps to matrix PA\n", name);

  uintptr_t input_pa = (uintptr_t)tlb_input_a;
  uintptr_t input_va = tlb_page_alias(TLB_ALIAS_BASE_4K, input_pa);

  tlb_setup_identity_4gb();
  tlb_map_4k(input_va, input_pa);
  tlb_enable_sv39();
  init_u8_random_matrix(tlb_input_a, TLB_DIM, TLB_DIM, 151);

  int passed = tlb_dma_roundtrip(input_va, tlb_input_a, tlb_output);
  return tlb_finish(name, passed);
}
