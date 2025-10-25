#include "isa.h"

// =========================== for simulator ===========================
const InstructionConfig mvin_config = {
    .rs1_fields = (BitFieldConfig[]){{"base_dram_addr", 0, 31}, {NULL, 0, 0}},
    .rs2_fields = (BitFieldConfig[]){{"col_stride", 24, 33},
                                     {"base_sp_addr", 0, 14},
                                     {"iter", 15, 24},
                                     {NULL, 0, 0}}};

// =========================== for CTest ===========================
#define MVIN_ENCODE_RS1(dram_addr) ENCODE_FIELD(dram_addr, 0, 32)

#define MVIN_ENCODE_RS2(sp_addr, iter, col_stride)                             \
  (ENCODE_FIELD(sp_addr, 0, 15) | ENCODE_FIELD(iter, 15, 10) |                 \
   ENCODE_FIELD(col_stride, 25, 10))

// MVIN指令低级实现
#ifndef __x86_64__
#define MVIN_RAW(rs1, rs2)                                                     \
  asm volatile(".insn r " STR(CUSTOM_3) ", 0x3, 24, x0, %0, %1"                \
               :                                                               \
               : "r"(rs1), "r"(rs2)                                            \
               : "memory")
#else
#define MVIN_RAW(rs1, rs2) /* x86平台下不执行RISC-V指令 */
#endif

// MVIN指令高级API实现
void bb_mvin(uint64_t mem_addr, uint32_t sp_addr, uint32_t iter,
             uint32_t col_stride) {
  uint32_t rs1_val = MVIN_ENCODE_RS1(mem_addr);
  uint32_t rs2_val = MVIN_ENCODE_RS2(sp_addr, iter, col_stride);
  MVIN_RAW(rs1_val, rs2_val);
}
