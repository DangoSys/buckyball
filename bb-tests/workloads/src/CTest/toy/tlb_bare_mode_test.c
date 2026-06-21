#include "tlb_common.h"

int main(void) {
  tlb_select_hart();
  const char *name = "TLB Bare Mode";
  printf("[%s] DMA with VM disabled\n", name);

  tlb_disable_vm();
  init_u8_random_matrix(tlb_input, TLB_DIM, TLB_DIM, 42);

  int passed = tlb_dma_roundtrip((uintptr_t)tlb_input, tlb_input, tlb_output);
  return tlb_finish(name, passed);
}
