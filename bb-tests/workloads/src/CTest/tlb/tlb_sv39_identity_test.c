#include "tlb_common.h"

int main(void) {
  tlb_select_hart();
  const char *name = "TLB Sv39 Identity";
  printf("[%s] DMA with Sv39 identity mapping\n", name);

  tlb_setup_identity_4gb();
  tlb_enable_sv39();
  init_u8_random_matrix(tlb_input, TLB_DIM, TLB_DIM, 77);

  int passed = tlb_dma_roundtrip((uintptr_t)tlb_input, tlb_input, tlb_output);
  return tlb_finish(name, passed);
}
