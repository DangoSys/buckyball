#ifndef BUCKYBALL_ISA_H
#define BUCKYBALL_ISA_H

#include <stddef.h>
#include <stdint.h>

// Data type for matrix elements
typedef int8_t elem_t;
typedef int32_t result_t;

// Custom instruction opcodes
#define CUSTOM_3 0x7b

// String macros
#define STR1(x) #x
#ifndef STR
#define STR(x) STR1(x)
#endif

// Field encoding macro with start and end bit
#define FIELD(val, start_bit, end_bit)                                         \
  (((val) & ((2UL << ((end_bit) - (start_bit))) - 1)) << (start_bit))

// Unified rs1 bank encoding flags (bits 45-47)
// bit 45 = rd_bank_0_valid, bit 46 = rd_bank_1_valid, bit 47 = wr_bank_valid
#define BB_RD0 (1UL << 45)
#define BB_RD1 (1UL << 46)
#define BB_WR (1UL << 47)

// rs1 bank field helpers (15-bit each)
#define BB_BANK0(id) FIELD(id, 0, 14)
#define BB_BANK1(id) FIELD(id, 15, 29)
#define BB_BANK2(id) FIELD(id, 30, 44)

// rs1 iter field (16-bit, bits 48-63)
#define BB_ITER(n) FIELD(n, 48, 63)

// Generic RISC-V custom instruction macro
#define BUCKYBALL_INSTRUCTION_R_R(rs1_val, rs2_val, func7)                     \
  asm volatile(".insn r " STR(CUSTOM_3) ", 0x3, %c2, x0, %0, %1"               \
               :                                                               \
               : "r"(rs1_val), "r"(rs2_val), "i"(func7)                        \
               : "memory")

// Include all instruction definitions
#include "21_shared_mvin.c"
#include "22_shared_mvout.c"
#include "23_mset.c"
#include "24_mvin.c"
#include "25_mvout.c"
#include "31_fence.c"
#include "32_mul_warp16.c"
#include "33_im2col.c"
#include "34_transpose.c"
#include "38_relu.c"
#include "39_bfp.c"
#include "40_quant.c"
#include "41_dequant.c"
#include "42_gemmini_config.c"
#include "43_gemmini_preload.c"
#include "44_gemmini_compute_preloaded.c"
#include "45_gemmini_compute_accumulated.c"
#include "46_gemmini_flush.c"

#endif // BUCKYBALL_ISA_H
