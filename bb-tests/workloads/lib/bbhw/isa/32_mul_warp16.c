#include "isa.h"

// =========================== for simulator ===========================
const InstructionConfig mul_warp16_config = {
    .rs1_fields = (BitFieldConfig[]){{"op1_spaddr", 0, 13},
                                     {"op2_spaddr", 14, 27},
                                     {NULL, 0, 0}},
    .rs2_fields = (BitFieldConfig[]){
        {"wr_spaddr", 0, 13}, {"iter", 14, 23}, {NULL, 0, 0}}};

// =========================== for CTest ===========================
#define MUL_WARP16_ENCODE_RS1(op1_addr, op2_addr)                              \
  (ENCODE_FIELD(op1_addr, 0, 14) | ENCODE_FIELD(op2_addr, 14, 14))

#define MUL_WARP16_ENCODE_RS2(wr_addr, iter)                                   \
  (ENCODE_FIELD(wr_addr, 0, 14) | ENCODE_FIELD(iter, 14, 10))

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
                   uint32_t iter) {
  uint32_t rs1_val = MUL_WARP16_ENCODE_RS1(op1_addr, op2_addr);
  uint32_t rs2_val = MUL_WARP16_ENCODE_RS2(wr_addr, iter);
  MUL_WARP16_RAW(rs1_val, rs2_val);
}
