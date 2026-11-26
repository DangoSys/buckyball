#ifndef BUCKYBALL_SPIKEIO_H
#define BUCKYBALL_SPIKEIO_H
#include <bbhw/isa/isa.h>

static inline uint64_t spikeio_banks(unsigned src, unsigned dst) {
  return BB_BANK0(src) | BB_BANK2(dst) | BB_ITER(1) | BB_UPDATE_WR;
}

// Bit i is set when signed sample i strictly exceeds the signed threshold.
static inline void spike_encode(unsigned src, unsigned dst, int8_t threshold) {
  BUCKYBALL_INSTRUCTION_R_R(spikeio_banks(src, dst), (uint64_t)(uint8_t)threshold,
                            FUNCT7(spikeio, SPIKE_ENCODE));
}

// Rows 0..7 hold signed 16-bit pairs. Return the lowest index attaining the maximum.
static inline void spike_argmax(unsigned src, unsigned dst) {
  BUCKYBALL_INSTRUCTION_R_R(spikeio_banks(src, dst), UINT64_C(0), FUNCT7(spikeio, SPIKE_ARGMAX));
}
#endif
