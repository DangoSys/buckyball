#include "tlb_common.h"

int main(void) {
  tlb_select_hart();
  const char *name = "TLB Sfence Flush";
  printf("[%s] sfence.vma flush + DMA refill\n", name);

  tlb_setup_identity_4gb();
  tlb_enable_sv39();
  asm volatile("sfence.vma" ::: "memory");
  init_u8_random_matrix(tlb_input, TLB_DIM, TLB_DIM, 99);

  int passed = tlb_dma_roundtrip((uintptr_t)tlb_input, tlb_input, tlb_output);
  return tlb_finish(name, passed);
}
