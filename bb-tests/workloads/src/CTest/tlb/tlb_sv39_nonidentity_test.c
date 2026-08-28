#include "tlb_common.h"

int main(void) {
  tlb_select_hart();
  const char *name = "TLB Sv39 Nonidentity";
  printf("[%s] 1GB VA alias maps to DRAM PA\n", name);

  uintptr_t input_pa = (uintptr_t)tlb_input;
  if (!tlb_pa_in_dram_1g(input_pa)) {
    printf("[%s] input PA outside expected DRAM window: 0x%lx\n", name,
           (unsigned long)input_pa);
    return tlb_finish(name, 0);
  }

  tlb_setup_identity_4gb();
  if (!tlb_map_1g(TLB_ALIAS_BASE_1G, TLB_DRAM_BASE)) {
    return tlb_finish(name, 0);
  }
  tlb_enable_sv39();
  init_u8_random_matrix(tlb_input, TLB_DIM, TLB_DIM, 123);

  uintptr_t input_va = tlb_dram_alias_1g(input_pa);
  int passed = tlb_dma_roundtrip(input_va, tlb_input, tlb_output);
  return tlb_finish(name, passed);
}
