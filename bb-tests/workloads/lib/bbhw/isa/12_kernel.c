#ifndef KERNEL_ISA_H
#define KERNEL_ISA_H

#include "isa.h"

struct kernel_launch {
  uint32_t entry;
  uint32_t end;
  uint32_t stack;
  uint32_t args[8];
  uint32_t reserved;
};

static inline void mvin_kernel(const void *image, uint32_t bytes,
                               uint32_t buffer) {
  BUCKYBALL_INSTRUCTION_R_R((uint64_t)bytes | ((uint64_t)buffer << 32),
                            (uintptr_t)image, 12);
}

static inline void run_kernel(uint32_t descriptor_bank_address,
                              uint32_t instruction_buffer) {
  BUCKYBALL_INSTRUCTION_R_R(descriptor_bank_address, instruction_buffer, 15);
}

#endif
