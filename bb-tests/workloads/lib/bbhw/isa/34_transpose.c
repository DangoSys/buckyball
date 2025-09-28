#include "isa.h"

// =========================== for simulator ===========================
const InstructionConfig transpose_config = {
    .rs1_fields = (BitFieldConfig[]){{"op_spaddr", 0, 13},
                                     {"wr_spaddr", 14, 27},
                                     {NULL, 0, 0}},
    .rs2_fields = (BitFieldConfig[]){{"iter", 14, 23}, {NULL, 0, 0}}};

// =========================== for CTest ===========================
#define TRANSPOSE_ENCODE_RS1(op_addr, wr_addr)                                 \
  (ENCODE_FIELD(op_addr, 0, 14) | ENCODE_FIELD(wr_addr, 14, 14))

#define TRANSPOSE_ENCODE_RS2(iter) ENCODE_FIELD(iter, 14, 10)

// TRANSPOSE指令低级实现
#ifndef __x86_64__
#define TRANSPOSE_RAW(rs1, rs2)                                                \
  asm volatile(".insn r " STR(CUSTOM_3) ", 0x3, 34, x0, %0, %1"                \
               :                                                               \
               : "r"(rs1), "r"(rs2)                                            \
               : "memory")
#else
#define TRANSPOSE_RAW(rs1, rs2) /* x86平台下不执行RISC-V指令 */
#endif

// TRANSPOSE指令高级API实现
void bb_transpose(uint32_t op1_addr, uint32_t wr_addr, uint32_t iter) {
  uint32_t rs1_val = TRANSPOSE_ENCODE_RS1(op1_addr, wr_addr);
  uint32_t rs2_val = TRANSPOSE_ENCODE_RS2(iter);
  TRANSPOSE_RAW(rs1_val, rs2_val);
}
