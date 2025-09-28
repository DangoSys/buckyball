#include "isa.h"

// =========================== for simulator ===========================
const InstructionConfig mvin_config = {
    .rs1_fields = (BitFieldConfig[]){{"base_dram_addr", 0, 31}, {NULL, 0, 0}},
    .rs2_fields = (BitFieldConfig[]){
        {"base_sp_addr", 0, 13}, {"iter", 14, 23}, {NULL, 0, 0}}};

// =========================== for CTest ===========================
#define MVIN_ENCODE_RS1(dram_addr) ENCODE_FIELD(dram_addr, 0, 32)

#define MVIN_ENCODE_RS2(sp_addr, iter)                                         \
  (ENCODE_FIELD(sp_addr, 0, 14) | ENCODE_FIELD(iter, 14, 10))

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
void bb_mvin(uint64_t mem_addr, uint32_t sp_addr, uint32_t iter) {
  uint32_t rs1_val = MVIN_ENCODE_RS1(mem_addr);
  uint32_t rs2_val = MVIN_ENCODE_RS2(sp_addr, iter);
  MVIN_RAW(rs1_val, rs2_val);
}
