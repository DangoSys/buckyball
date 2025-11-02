#include "isa.h"

// =========================== for simulator ===========================
const InstructionConfig bbfp_mul_config = {
    .rs1_fields = (BitFieldConfig[]){{"op1_spaddr", 0, 14},
                                     {"op2_spaddr", 15, 29},
                                     {NULL, 0, 0}},
    .rs2_fields = (BitFieldConfig[]){
        {"wr_spaddr", 0, 14}, {"iter", 15, 24}, {NULL, 0, 0}}};

// =========================== for CTest ===========================
#define BBFP_MUL_ENCODE_RS1(op1_addr, op2_addr)                                \
  (ENCODE_FIELD(op1_addr, 0, 15) | ENCODE_FIELD(op2_addr, 15, 15))

#define BBFP_MUL_ENCODE_RS2(wr_addr, iter)                                     \
  (ENCODE_FIELD(wr_addr, 0, 15) | ENCODE_FIELD(iter, 15, 10))

// BBFP_MUL instruction low-level implementation
#ifndef __x86_64__
#define BBFP_MUL_RAW(rs1, rs2)                                                 \
  asm volatile(".insn r " STR(CUSTOM_3) ", 0x3, 26, x0, %0, %1"                \
               :                                                               \
               : "r"(rs1), "r"(rs2)                                            \
               : "memory")
#else
// Do not execute RISC-V instructions on x86 platform
#define BBFP_MUL_RAW(rs1, rs2)
#endif

// BBFP_MUL instruction high-level API implementation
void bb_bbfp_mul(uint32_t op1_addr, uint32_t op2_addr, uint32_t wr_addr,
                 uint32_t iter) {
  uint64_t rs1_val = BBFP_MUL_ENCODE_RS1(op1_addr, op2_addr);
  uint64_t rs2_val = BBFP_MUL_ENCODE_RS2(wr_addr, iter);
  BBFP_MUL_RAW(rs1_val, rs2_val);
}
