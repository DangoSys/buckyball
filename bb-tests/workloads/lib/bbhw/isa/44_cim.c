#include "isa.h"

// =========================== for simulator ===========================
const InstructionConfig cim_config = {
    .rs1_fields = (BitFieldConfig[]){{"op1_spaddr", 0, 14},
                                     {"op2_spaddr", 15, 29},
                                     {NULL, 0, 0}},
    .rs2_fields = (BitFieldConfig[]){
        {"result_spaddr", 0, 14},
        {"iter", 15, 24},
        // special field is 40 bits: rs2(63, spAddrLen + 10) = rs2(63, 24) = 40
        // bits Encode: rows[15:0], cols[31:16], op_type[35:32]
        {"rows", 25, 40},
        {"cols", 41, 56},
        {"op_type", 57, 60},
        {NULL, 0, 0}}};

// =========================== for CTest ===========================
#define CIM_ENCODE_RS1(op1_addr, op2_addr)                                     \
  (ENCODE_FIELD(op1_addr, 0, 15) | ENCODE_FIELD(op2_addr, 15, 15))

// Note: special field is only 40 bits (rs2[63:24])
// DomainDecoder extracts rs2(63, spAddrLen + 10) = rs2(63, 24) for special
// So special[39:0] = rs2[63:24]
// Encode in special: rows[15:0] = special[15:0], cols[15:0] = special[31:16],
// op_type[3:0] = special[35:32] In rs2: rows in rs2[39:24], cols in rs2[55:40],
// op_type in rs2[59:56]
#define CIM_ENCODE_RS2(result_addr, iter, rows, cols, op_type)                 \
  (ENCODE_FIELD(result_addr, 0, 15) | ENCODE_FIELD(iter, 15, 10) |             \
   ENCODE_FIELD(rows, 24, 16) | ENCODE_FIELD(cols, 40, 16) |                   \
   ENCODE_FIELD(op_type, 56, 4))

// CIM instruction low-level implementation
#ifndef __x86_64__
#define CIM_RAW(rs1, rs2)                                                      \
  asm volatile(".insn r " STR(CUSTOM_3) ", 0x3, 44, x0, %0, %1"                \
               :                                                               \
               : "r"(rs1), "r"(rs2)                                            \
               : "memory")
#else
// Do not execute RISC-V instructions on x86 platform
#define CIM_RAW(rs1, rs2)
#endif

// CIM instruction high-level API implementation
void bb_cim(uint32_t op1_addr, uint32_t op2_addr, uint32_t result_addr,
            uint32_t iter, uint32_t rows, uint32_t cols, uint32_t op_type) {
  uint64_t rs1_val = CIM_ENCODE_RS1(op1_addr, op2_addr);
  uint64_t rs2_val = CIM_ENCODE_RS2(result_addr, iter, rows, cols, op_type);
  CIM_RAW(rs1_val, rs2_val);
}
