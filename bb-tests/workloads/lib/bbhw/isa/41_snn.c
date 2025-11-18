#include "isa.h"

// =========================== for simulator ===========================
const InstructionConfig snn_config = {
    .rs1_fields = (BitFieldConfig[]){{"op_spaddr", 0, 14}, {NULL, 0, 0}},
    .rs2_fields = (BitFieldConfig[]){{"wr_spaddr", 0, 14},
                                     {"iter", 15, 24},
                                     {"threshold", 25, 32},
                                     {"leak_factor", 33, 40},
                                     {NULL, 0, 0}}};

// =========================== for CTest ===========================
#define SNN_ENCODE_RS1(op_addr) (ENCODE_FIELD(op_addr, 0, 15))
#define SNN_ENCODE_RS2(wr_addr, iter, threshold, leak_factor)                  \
  (ENCODE_FIELD(wr_addr, 0, 15) | ENCODE_FIELD(iter, 15, 10) |                 \
   ENCODE_FIELD(threshold, 25, 8) | ENCODE_FIELD(leak_factor, 33, 8))

// SNN instruction low-level implementation
#ifndef __x86_64__
#define SNN_RAW(rs1, rs2)                                                      \
  asm volatile(".insn r " STR(CUSTOM_3) ", 0x3, 41, x0, %0, %1"                \
               :                                                               \
               : "r"(rs1), "r"(rs2)                                            \
               : "memory")
#else
// Do not execute RISC-V instructions on x86 platform
#define SNN_RAW(rs1, rs2)
#endif

// SNN instruction high-level API implementation
void bb_snn(uint32_t op1_addr, uint32_t wr_addr, uint32_t iter,
            uint32_t threshold, uint32_t leak_factor) {
  uint64_t rs1_val = SNN_ENCODE_RS1(op1_addr);
  uint64_t rs2_val = SNN_ENCODE_RS2(wr_addr, iter, threshold, leak_factor);
  SNN_RAW(rs1_val, rs2_val);
}
