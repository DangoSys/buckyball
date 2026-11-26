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
static inline void syntzulu_run_bank_delayed(unsigned input, unsigned output, unsigned delay_steps) {
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(input) | syntzulu_destination(output), (uint64_t)(delay_steps & 3),
                            FUNCT7(syntzulu, SYNTZULU_RUN_BANK));
}

// Identical immediate-frame submission and eight-row output for both weight modes.
static inline void syntzulu_run_frame(unsigned weight_bank, unsigned output,
                                     uint16_t frame, unsigned delay_steps,
                                     unsigned shared_weights) {
  uint64_t args = ((uint64_t)frame << 16) | 8u |
                  ((uint64_t)(shared_weights & 1) << 2) | (delay_steps & 3);
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(weight_bank) | syntzulu_destination(output),
                            args, FUNCT7(syntzulu, SYNTZULU_RUN_BANK));
}

// Source holds 128 packed 64-bit weight words, two per 128-bit SRAM row.
static inline void syntzulu_run_shared(unsigned weight_bank, unsigned output, uint16_t frame, unsigned delay_steps) {
  uint64_t args = ((uint64_t)frame << 16) | 4u | (delay_steps & 3);
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(weight_bank) | syntzulu_destination(output), args,
                            FUNCT7(syntzulu, SYNTZULU_RUN_BANK));
}

static inline void syntzulu_run_bank(unsigned input, unsigned output) {
  syntzulu_run_bank_delayed(input, output, 0);
}

#endif

// READ selector 8 writes the last output spike mask to row zero of bank.
static inline void syntzulu_store_spikes(unsigned bank) {
  BUCKYBALL_INSTRUCTION_R_R(syntzulu_destination(bank), UINT64_C(8), FUNCT7(syntzulu, SYNTZULU_READ));
}

static inline uint32_t syntzulu_read(unsigned bank, unsigned index, volatile uint32_t *row) {
  BUCKYBALL_INSTRUCTION_R_R(syntzulu_destination(bank), (uint64_t)(index & 7), FUNCT7(syntzulu, SYNTZULU_READ));
  bb_mvout((uintptr_t)row, bank, 1, 1);
  asm volatile("fence rw,rw" ::: "memory");
  return row[0];
}
#endif
