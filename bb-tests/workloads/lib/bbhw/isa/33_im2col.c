#include "isa.h"

// =========================== for simulator ===========================
const InstructionConfig im2col_config = {
    .rs1_fields = (BitFieldConfig[]){{"op_spaddr", 0, 14},
                                     {"wr_spaddr", 15, 29},
                                     {NULL, 0, 0}},
    .rs2_fields = (BitFieldConfig[]){{"kcol", 26, 29},
                                     {"krow", 30, 33},
                                     {"incol", 34, 38},
                                     {"inrow", 39, 43},
                                     {"startcol", 49, 53},
                                     {"startrow", 54, 58},
                                     {NULL, 0, 0}}};

// =========================== for CTest ===========================
#define IM2COL_ENCODE_RS1(op_addr, wr_addr)                                    \
  (ENCODE_FIELD(op_addr, 0, 15) | ENCODE_FIELD(wr_addr, 15, 15))

#define IM2COL_ENCODE_RS2(krow, kcol, inrow, incol, startrow, startcol)        \
  (ENCODE_FIELD(kcol, 26, 4) | ENCODE_FIELD(krow, 30, 4) |                     \
   ENCODE_FIELD(incol, 34, 5) | ENCODE_FIELD(inrow, 39, 5) |                   \
   ENCODE_FIELD(startcol, 49, 5) | ENCODE_FIELD(startrow, 54, 5))

// IM2COL指令低级实现
#ifndef __x86_64__
#define IM2COL_RAW(rs1, rs2)                                                   \
  asm volatile(".insn r " STR(CUSTOM_3) ", 0x3, 33, x0, %0, %1"                \
               :                                                               \
               : "r"(rs1), "r"(rs2)                                            \
               : "memory")
#else
#define IM2COL_RAW(rs1, rs2) /* x86平台下不执行RISC-V指令 */
#endif

// IM2COL指令高级API实现
void bb_im2col(uint32_t op1_addr, uint32_t wr_addr, uint32_t krow,
               uint32_t kcol, uint32_t inrow, uint32_t incol, uint32_t startrow,
               uint32_t startcol) {
  uint64_t rs1_val = IM2COL_ENCODE_RS1(op1_addr, wr_addr);
  uint64_t rs2_val =
      IM2COL_ENCODE_RS2(krow, kcol, inrow, incol, startrow, startcol);
  IM2COL_RAW(rs1_val, rs2_val);
}
