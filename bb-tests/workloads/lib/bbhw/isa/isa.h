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

// rs1 bank field helpers (10-bit each)
#define BB_BANK0(id) FIELD(id, 0, 9)
#define BB_BANK1(id) FIELD(id, 10, 19)
#define BB_BANK2(id) FIELD(id, 20, 29)

// rs1 iter field (34-bit, bits 30-63)
#define BB_ITER(n) FIELD(n, 30, 63)

// funct7 encoding: [6:4]=enable, [3:0]=opcode
// enable: 000=none, 001=1rd, 010=1wr, 011=1rd+1wr, 100=2rd+1wr
//         101/110/111 = none (extended opcode space)

// Generic RISC-V custom instruction macro (funct3 always 0x3 = CUSTOM3_RS1_RS2)
#define BUCKYBALL_INSTRUCTION_R_R(rs1_val, rs2_val, func7)                     \
  asm volatile(".insn r " STR(CUSTOM_3) ", 3, %c2, x0, %0, %1"                 \
               :                                                               \
               : "r"(rs1_val), "r"(rs2_val), "i"(func7)                        \
               : "memory")

// Include all instruction definitions
#include "00_fence.c"
#include "01_barrier.c"
#include "02_gemmini_config.c"
#include "03_gemmini_flush.c"
#include "04_bdb_counter.c"
#include "16_mvout.c"
#include "32_mset.c"
#include "33_mvin.c"
#include "48_im2col.c"
#include "49_transpose.c"
#include "50_relu.c"
#include "51_quant.c"
#include "52_dequant.c"
#include "53_gemmini_preload.c"
#include "64_mul_warp16.c"
#include "65_bfp.c"
#include "66_gemmini_compute_preloaded.c"
#include "67_gemmini_compute_accumulated.c"
#include "80_gemmini_loop_ws.c"
#include "96_gemmini_loop_conv_ws.c"

#endif // BUCKYBALL_ISA_H
