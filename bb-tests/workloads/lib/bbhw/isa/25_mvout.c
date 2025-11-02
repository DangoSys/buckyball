#include "isa.h"

// =========================== for simulator ===========================
const InstructionConfig mvout_config = {
    .rs1_fields = (BitFieldConfig[]){{"base_dram_addr", 0, 31}, {NULL, 0, 0}},
    .rs2_fields = (BitFieldConfig[]){{"base_sp_addr", 0, 14},
                                     {"iter", 15, 24},
                                     {"stride", 24, 33},
                                     {NULL, 0, 0}}};

// =========================== for CTest ===========================
#define MVOUT_ENCODE_RS1(dram_addr) ENCODE_FIELD(dram_addr, 0, 32)

#define MVOUT_ENCODE_RS2(sp_addr, iter, stride)                                \
  (ENCODE_FIELD(sp_addr, 0, 15) | ENCODE_FIELD(iter, 15, 10) |                 \
   ENCODE_FIELD(stride, 25, 10))

// MVOUT instruction low-level implementation
#ifndef __x86_64__
#define MVOUT_RAW(rs1, rs2)                                                    \
  asm volatile(".insn r " STR(CUSTOM_3) ", 0x3, 25, x0, %0, %1"                \
               :                                                               \
               : "r"(rs1), "r"(rs2)                                            \
               : "memory")
#else
// Do not execute RISC-V instructions on x86 platform
#define MVOUT_RAW(rs1, rs2)
#endif

// MVOUT instruction high-level API implementation
void bb_mvout(uint64_t mem_addr, uint32_t sp_addr, uint32_t iter,
              uint32_t stride) {
  uint64_t rs1_val = MVOUT_ENCODE_RS1(mem_addr);
  uint64_t rs2_val = MVOUT_ENCODE_RS2(sp_addr, iter, stride);
  MVOUT_RAW(rs1_val, rs2_val);
}
