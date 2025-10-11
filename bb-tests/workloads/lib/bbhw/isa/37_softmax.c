#include "isa.h"

// =========================== for simulator ===========================
const InstructionConfig softmax_config = {
    .rs1_fields = (BitFieldConfig[]){{"op1_bank", 0, 1},
                                     {"op1_spaddr", 2, 13},
                                     {"wr_bank", 14, 15},
                                     {"wr_spaddr", 16, 27},
                                     {NULL, 0, 0}},
    .rs2_fields = (BitFieldConfig[]){{"iter", 0, 9},
                                     {"is_acc", 10, 10},
                                     {"dim_len", 11, 20},
                                     {"batch", 21, 30},
                                     {"log_mode", 31, 31},
                                     {NULL, 0, 0}}};

// =========================== for CTest ===========================
// RS1: [op1_bank(2) | op1_addr(14) | wr_bank(2) | wr_addr(14)]
#define SOFTMAX_ENCODE_RS1(op1_bank, op1_addr, wr_bank, wr_addr)               \
  (ENCODE_FIELD(op1_bank, 0, 2) | ENCODE_FIELD(op1_addr, 2, 14) |              \
   ENCODE_FIELD(wr_bank, 16, 2) | ENCODE_FIELD(wr_addr, 18, 14))

// RS2: [iter(10) | is_acc(1) | dim_len(10) | batch(10) | log_mode(1)]
#define SOFTMAX_ENCODE_RS2(iter, is_acc, dim_len, batch, log_mode)             \
  (ENCODE_FIELD(iter, 0, 10) | ENCODE_FIELD(is_acc, 10, 1) |                   \
   ENCODE_FIELD(dim_len, 11, 10) | ENCODE_FIELD(batch, 21, 10) |               \
   ENCODE_FIELD(log_mode, 31, 1))

// Softmax指令低级实现
#ifndef __x86_64__
#define SOFTMAX_RAW(rs1, rs2)                                                  \
  asm volatile(".insn r " STR(CUSTOM_3) ", 0x3, 37, x0, %0, %1"                \
               :                                                               \
               : "r"(rs1), "r"(rs2)                                            \
               : "memory")
#else
#define SOFTMAX_RAW(rs1, rs2) /* x86平台下不执行RISC-V指令 */
#endif

// Softmax指令高级API实现
void bb_softmax(uint32_t op1_bank, uint32_t op1_addr, uint32_t wr_bank,
                uint32_t wr_addr, uint32_t iter, uint32_t is_acc,
                uint32_t dim_len, uint32_t batch, uint32_t log_mode) {
  uint64_t rs1_val = SOFTMAX_ENCODE_RS1(op1_bank, op1_addr, wr_bank, wr_addr);
  uint64_t rs2_val = SOFTMAX_ENCODE_RS2(iter, is_acc, dim_len, batch, log_mode);
  SOFTMAX_RAW(rs1_val, rs2_val);
}
