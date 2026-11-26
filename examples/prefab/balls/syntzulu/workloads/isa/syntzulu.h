#ifndef BUCKYBALL_SYNTZULU_H
#define BUCKYBALL_SYNTZULU_H
#include <bbhw/isa/isa.h>

static inline uint64_t syntzulu_destination(unsigned bank) {
  return BB_BANK2(bank) | BB_ITER(1) | BB_UPDATE_WR;
}

static inline void syntzulu_weight(unsigned bank, unsigned half, unsigned address, uint32_t pair) {
  uint64_t data = ((uint64_t)(half & 1) << 39) | ((uint64_t)(address & 127) << 32) | pair;
  BUCKYBALL_INSTRUCTION_R_R(syntzulu_destination(bank), data, FUNCT7(syntzulu, SYNTZULU_WEIGHT));
  asm volatile("fence rw,rw" ::: "memory");
}

static inline void syntzulu_run(unsigned bank, uint16_t frame) {
  BUCKYBALL_INSTRUCTION_R_R(syntzulu_destination(bank), (uint64_t)frame, FUNCT7(syntzulu, SYNTZULU_RUN));
  asm volatile("fence rw,rw" ::: "memory");
}

#ifdef THREAD_BALLDOMAIN_BALLS_syntzulu_INSTRUCTIONS_SYNTZULU_RUN_BANK_FUNCT7
// One packed frame in row zero of input; eight result pairs in output rows 0..7.
static inline void syntzulu_run_bank(unsigned input, unsigned output) {
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(input) | syntzulu_destination(output), UINT64_C(0),
                            FUNCT7(syntzulu, SYNTZULU_RUN_BANK));
}

#endif

static inline uint32_t syntzulu_read(unsigned bank, unsigned index, volatile uint32_t *row) {
  BUCKYBALL_INSTRUCTION_R_R(syntzulu_destination(bank), (uint64_t)(index & 7), FUNCT7(syntzulu, SYNTZULU_READ));
  bb_mvout((uintptr_t)row, bank, 1, 1);
  asm volatile("fence rw,rw" ::: "memory");
  return row[0];
}
#endif
