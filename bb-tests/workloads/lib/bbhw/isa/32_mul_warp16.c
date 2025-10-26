#include "isa.h"

// =========================== for simulator ===========================
const InstructionConfig mul_warp16_config = {
    .rs1_fields = (BitFieldConfig[]){{"op1_spaddr", 0, 14},
                                     {"op2_spaddr", 15, 29},
                                     {NULL, 0, 0}},
    .rs2_fields = (BitFieldConfig[]){{"wr_spaddr", 0, 14},
                                     {"iter", 15, 24},
                                     {"mode", 25, 25},
                                     {NULL, 0, 0}}};

// =========================== for CTest ===========================
#define MUL_WARP16_ENCODE_RS1(op1_addr, op2_addr)                              \
  (ENCODE_FIELD(op1_addr, 0, 15) | ENCODE_FIELD(op2_addr, 15, 15))

#define MUL_WARP16_ENCODE_RS2(wr_addr, iter, mode)                             \
  (ENCODE_FIELD(wr_addr, 0, 15) | ENCODE_FIELD(iter, 15, 10) |                 \
   ENCODE_FIELD(mode, 25, 1))

// MUL_WARP16指令低级实现
#ifndef __x86_64__
#define MUL_WARP16_RAW(rs1, rs2)                                               \
  asm volatile(".insn r " STR(CUSTOM_3) ", 0x3, 32, x0, %0, %1"                \
               :                                                               \
               : "r"(rs1), "r"(rs2)                                            \
               : "memory")
#else
#define MUL_WARP16_RAW(rs1, rs2) /* x86平台下不执行RISC-V指令 */
#endif

// MUL_WARP16指令高级API实现
void bb_mul_warp16(uint32_t op1_addr, uint32_t op2_addr, uint32_t wr_addr,
                   uint32_t iter, uint32_t mode) {
  uint64_t rs1_val = MUL_WARP16_ENCODE_RS1(op1_addr, op2_addr);
  uint64_t rs2_val = MUL_WARP16_ENCODE_RS2(wr_addr, iter, mode);
  MUL_WARP16_RAW(rs1_val, rs2_val);
}
