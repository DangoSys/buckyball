#include "isa.h"

// =========================== for simulator ===========================
const InstructionConfig transpose_config = {
    .rs1_fields = (BitFieldConfig[]){{"op_spaddr", 0, 14},
                                     {"wr_spaddr", 15, 29},
                                     {NULL, 0, 0}},
    .rs2_fields =
        (BitFieldConfig[]){{"mode", 25, 25}, {"iter", 15, 24}, {NULL, 0, 0}}};

// =========================== for CTest ===========================
#define TRANSPOSE_ENCODE_RS1(op_addr, wr_addr)                                 \
  (ENCODE_FIELD(op_addr, 0, 15) | ENCODE_FIELD(wr_addr, 15, 15))

#define TRANSPOSE_ENCODE_RS2(iter, mode)                                       \
  (ENCODE_FIELD(iter, 15, 10) | ENCODE_FIELD(mode, 25, 1))

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
void bb_transpose(uint32_t op1_addr, uint32_t wr_addr, uint32_t iter,
                  uint32_t mode) {
  uint64_t rs1_val = TRANSPOSE_ENCODE_RS1(op1_addr, wr_addr);
  uint64_t rs2_val = TRANSPOSE_ENCODE_RS2(iter, mode);
  TRANSPOSE_RAW(rs1_val, rs2_val);
}
