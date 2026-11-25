#ifndef OPERATOR_ISA_H
#define OPERATOR_ISA_H

#include "isa.h"

struct operator_launch {
  uint64_t entry;
  uint64_t end;
  uint64_t stack;
  uint64_t args[8];
  uint64_t reserved;
};

static inline void mvin_operator(const void *image, uint32_t bytes,
                                 uint32_t program_bank) {
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK2(program_bank) | BB_ITER(bytes),
                            (uintptr_t)image, 44);
}

static inline void run_operator(uint32_t read_bank, uint32_t program_bank,
                                uint32_t write_bank,
                                uint32_t descriptor_offset) {
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(read_bank) | BB_BANK1(program_bank) |
                                BB_BANK2(write_bank),
                            descriptor_offset, 79);
}

static inline void release_operator(uint32_t program_bank) {
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(program_bank), 0, 32);
}

#endif
