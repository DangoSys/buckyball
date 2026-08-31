#ifndef BUCKYBALL_ISA_H
#define BUCKYBALL_ISA_H

#include <stddef.h>
#include <stdint.h>

#if defined(BUCKYBALL_RUSHB)
#include <buckyball/rushb.h>
#if !defined(BUCKYBALL_RUSHB_CORE_ID)
#error "RushB workloads require a Core ID"
#endif
#define BUCKYBALL_RUSHB_CORE (uint32_t)BUCKYBALL_RUSHB_CORE_ID
#endif

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
#if defined(BUCKYBALL_RUSHB)
#define BUCKYBALL_INSTRUCTION_R_R(rs1_val, rs2_val, func7)                     \
  do {                                                                         \
    rushb_custom(BUCKYBALL_RUSHB_CORE, (uint64_t)(rs1_val),                    \
                 (uint64_t)(rs2_val), (uint32_t)(func7));                      \
  } while (0)
#else
#define BUCKYBALL_INSTRUCTION_R_R(rs1_val, rs2_val, func7)                     \
  asm volatile(".insn r " STR(CUSTOM_3) ", 3, %c2, x0, %0, %1"                 \
               :                                                               \
               : "r"(rs1_val), "r"(rs2_val), "i"(func7)                        \
               : "memory")
#endif

// Base (mem/frontend) instruction definitions only.
// Ball-specific ISA macros live under examples/balls/<ball>/workloads/isa/.
#include "00_fence.c"
#include "01_barrier.c"
#include "16_mvout.c"
#include "32_mset.c"
#include "33_mvin.c"
#include "35_mvin_mmio.c"

#endif // BUCKYBALL_ISA_H
