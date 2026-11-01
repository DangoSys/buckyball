#ifndef BUCKYBALL_BB_ISA_H
#define BUCKYBALL_BB_ISA_H

// The Buckyball instruction encodings of the C API.

#include <stdint.h>

// Buckyball instruction encoding: RISC-V custom-3, funct3 = 3 (rs1 and rs2),
// funct7 selects the instruction.
#define CUSTOM_3 0x7b

#define STR1(x) #x
#ifndef STR
#define STR(x) STR1(x)
#endif

// Bits [start_bit, end_bit] of an operand.
#define FIELD(val, start_bit, end_bit)                                         \
  (((val) & ((2UL << ((end_bit) - (start_bit))) - 1)) << (start_bit))

// rs1 bank fields: 10-bit sources and 9-bit architectural destination.
#define BB_BANK0(id) FIELD(id, 0, 9)
#define BB_BANK1(id) FIELD(id, 10, 19)
#define BB_BANK2(id) FIELD(id, 20, 28)
// The architectural destination also supplies an old-value read operand.
#define BB_READ_WR_OLD (1ULL << 29)
#define BB_UPDATE_WR (1ULL << 63)

// rs1 iter field (33-bit, bits 30-62); bit 63 declares an in-place update.
#define BB_ITER(n) FIELD(n, 30, 62)

// funct7 of the base (memory and frontend) instructions.
#define BB_BARRIER_FUNC7 1
#define BB_MVOVER_FUNC7 0x0d
#define BB_MVOUT_FUNC7 16
#define BB_MSET_FUNC7 32
#define BB_MVIN_FUNC7 33
#define BB_MVIN_2D_FUNC7 34
#define BB_MVIN_MMIO_FUNC7 35

// The funct7 of instruction `mne` of the Ball in directory `ball`, from the
// compile definitions of the target (DESIGN_TARGET_DEFS in design.cmake).
#define FUNCT7(ball, mne)                                                      \
  THREAD_BALLDOMAIN_BALLS_##ball##_INSTRUCTIONS_##mne##_FUNCT7

// Issues one instruction on the calling hart.
#define BUCKYBALL_INSTRUCTION_R_R(rs1_val, rs2_val, func7)                     \
  asm volatile(".insn r " STR(CUSTOM_3) ", 3, %c2, x0, %0, %1"                 \
               :                                                               \
               : "r"(rs1_val), "r"(rs2_val), "i"(func7)                        \
               : "memory")

#endif // BUCKYBALL_BB_ISA_H
