#include "tlb_common.h"

int main(void) {
  tlb_select_hart();
  const char *name = "TLB Repeated Sfence";
  printf("[%s] repeated sfence.vma + DMA cycles\n", name);

  tlb_setup_identity_4gb();
  tlb_enable_sv39();

  int passed = 1;
  for (int i = 0; i < 4; i++) {
    asm volatile("sfence.vma" ::: "memory");
    init_u8_random_matrix(tlb_input, TLB_DIM, TLB_DIM, 100 + i);
    if (!tlb_dma_roundtrip((uintptr_t)tlb_input, tlb_input, tlb_output)) {
      printf("[%s] iteration %d mismatch\n", name, i);
      passed = 0;
      break;
    }
  }

  return tlb_finish(name, passed);
}
