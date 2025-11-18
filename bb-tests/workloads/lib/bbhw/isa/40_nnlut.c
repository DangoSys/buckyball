#include "isa.h"

// =========================== for simulator ===========================
const InstructionConfig nnlut_config = {
    .rs1_fields = (BitFieldConfig[]){{"op_spaddr", 0, 14}, {NULL, 0, 0}},
    .rs2_fields = (BitFieldConfig[]){
        {"wr_spaddr", 0, 14}, {"iter", 15, 24}, {NULL, 0, 0}}};

// =========================== for CTest ===========================
#define NNLUT_ENCODE_RS1(op_addr) (ENCODE_FIELD(op_addr, 0, 15))
#define NNLUT_ENCODE_RS2(wr_addr, iter)                                        \
  (ENCODE_FIELD(wr_addr, 0, 15) | ENCODE_FIELD(iter, 15, 10))

// NNLUT instruction low-level implementation
#ifndef __x86_64__
#define NNLUT_RAW(rs1, rs2)                                                    \
  asm volatile(".insn r " STR(CUSTOM_3) ", 0x3, 40, x0, %0, %1"                \
               :                                                               \
               : "r"(rs1), "r"(rs2)                                            \
               : "memory")
#else
// Do not execute RISC-V instructions on x86 platform
#define NNLUT_RAW(rs1, rs2)
#endif

// NNLUT instruction high-level API implementation
void bb_nnlut(uint32_t op1_addr, uint32_t wr_addr, uint32_t iter) {
  uint64_t rs1_val = NNLUT_ENCODE_RS1(op1_addr);
  uint64_t rs2_val = NNLUT_ENCODE_RS2(wr_addr, iter);
  NNLUT_RAW(rs1_val, rs2_val);
}
