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
  (((val) & ((1UL << ((end_bit) - (start_bit) + 1)) - 1)) << (start_bit))

// Generic RISC-V custom instruction macro
#define BUCKYBALL_INSTRUCTION_R_R(rs1_val, rs2_val, func7)                     \
  asm volatile(".insn r " STR(CUSTOM_3) ", 0x3, %c2, x0, %0, %1"               \
               :                                                               \
               : "r"(rs1_val), "r"(rs2_val), "i"(func7)                        \
               : "memory")

// Include all instruction definitions
#include "23_mset.c"
#include "24_mvin.c"
#include "25_mvout.c"
#include "26_bbfp_mul.c"
#include "27_matmul_ws.c"
#include "31_fence.c"
#include "32_mul_warp16.c"
#include "33_im2col.c"
#include "34_transpose.c"
#include "38_relu.c"
#include "39_bbus_config.c"
#include "40_nnlut.c"
#include "41_snn.c"
#include "42_abft_systolic.c"
#include "43_conv.c"
#include "44_cim.c"
#include "45_transfer.c"
#include "7_flush.c"

#endif // BUCKYBALL_ISA_H
