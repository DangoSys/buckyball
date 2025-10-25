#include "isa.h"

// =========================== for simulator ===========================
const InstructionConfig matmul_ws_config = {
    .rs1_fields = (BitFieldConfig[]){{"op1_spaddr", 0, 14},
                                     {"op2_spaddr", 15, 29},
                                     {NULL, 0, 0}},
    .rs2_fields = (BitFieldConfig[]){{"wr_spaddr", 0, 14},
                                     {"iter", 15, 24},
                                     {"ws_flag", 25, 25},
                                     {NULL, 0, 0}}};

// =========================== for CTest ===========================
#define MATMUL_WS_ENCODE_RS1(op1_addr, op2_addr)                               \
  (ENCODE_FIELD(op1_addr, 0, 15) | ENCODE_FIELD(op2_addr, 15, 15))

#define MATMUL_WS_ENCODE_RS2(wr_addr, iter, ws_flag)                           \
  (ENCODE_FIELD(wr_addr, 0, 15) | ENCODE_FIELD(iter, 15, 10) |                 \
   ENCODE_FIELD(ws_flag, 25, 1))

// MATMUL_WS指令低级实现
#ifndef __x86_64__
#define MATMUL_WS_RAW(rs1, rs2)                                                \
  asm volatile(".insn r " STR(CUSTOM_3) ", 0x3, 27, x0, %0, %1"                \
               :                                                               \
               : "r"(rs1), "r"(rs2)                                            \
               : "memory")
#else
#define MATMUL_WS_RAW(rs1, rs2) /* x86平台下不执行RISC-V指令 */
#endif

// MATMUL_WS指令高级API实现
void bb_matmul_ws(uint32_t op1_addr, uint32_t op2_addr, uint32_t wr_addr,
                  uint32_t iter) {
  uint32_t rs1_val = MATMUL_WS_ENCODE_RS1(op1_addr, op2_addr);
  uint32_t rs2_val = MATMUL_WS_ENCODE_RS2(wr_addr, iter, 1);
  MATMUL_WS_RAW(rs1_val, rs2_val);
}
