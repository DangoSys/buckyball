#include "isa.h"
#include <stdio.h>
#include <stdlib.h>
// =========================== for simulator ===========================
const InstructionConfig bbus_config_config = {
    .rs1_fields = (BitFieldConfig[]){{NULL, 0, 0}},
    .rs2_fields = (BitFieldConfig[]){{"src_bid", 25, 30},
                                     {"dst_bid", 31, 36},
                                     {"enable", 37, 37},
                                     {NULL, 0, 0}}};

// =========================== for CTest ===========================
#define bbus_config_ENCODE_RS1(op_addr) (0)
#define bbus_config_ENCODE_RS2(src_bid, dst_bid, enable)                       \
  (ENCODE_FIELD(src_bid, 25, 6) | ENCODE_FIELD(dst_bid, 31, 6) |               \
   ENCODE_FIELD(enable, 37, 1))

// bbus_config instruction low-level implementation
#ifndef __x86_64__
#define bbus_config_RAW(rs1, rs2)                                              \
  asm volatile(".insn r " STR(CUSTOM_3) ", 0x3, 39, x0, %0, %1"                \
               :                                                               \
               : "r"(rs1), "r"(rs2)                                            \
               : "memory")
#else
// Do not execute RISC-V instructions on x86 platform
#define bbus_config_RAW(rs1, rs2)
#endif

// bbus_config instruction high-level API implementation
void bb_bbus_config(uint32_t src_bid, uint32_t dst_bid, uint64_t enable) {
  uint64_t rs1_val = bbus_config_ENCODE_RS1(0);
  uint64_t rs2_val = bbus_config_ENCODE_RS2(src_bid, dst_bid, enable);
  bbus_config_RAW(rs1_val, rs2_val);
}
