#include "isa.h"

// IM2COL指令配置
const InstructionConfig im2col_config = {
    .rs1_fields = (BitFieldConfig[]){{"op_spaddr", 0, 13},
                                     {"wr_spaddr", 14, 27},
                                     {NULL, 0, 0}},
    .rs2_fields = (BitFieldConfig[]){{"kcol", 26, 29},
                                     {"krow", 30, 33},
                                     {"incol", 34, 38},
                                     {"inrow", 39, 43},
                                     {"startcol", 49, 53},
                                     {"startrow", 54, 58},
                                     {NULL, 0, 0}}};

// IM2COL指令执行函数
#ifndef __x86_64__
static void execute_im2col_impl(uint32_t rs1_val, uint32_t rs2_val) {
  asm volatile(".insn r " STR(CUSTOM_3) ", 0x3, 33, x0, %0, %1"
               : : "r"(rs1_val), "r"(rs2_val) : "memory");
}
#else
static void execute_im2col_impl(uint32_t rs1_val, uint32_t rs2_val) {
  // x86平台下不执行RISC-V指令
}
#endif

// 注册IM2COL指令
void register_im2col_instruction(void) {
  register_instruction(IM2COL_FUNC7, execute_im2col_impl);
}

// IM2COL指令高级API实现
void bb_im2col(uint32_t op1_addr, uint32_t wr_addr, uint32_t krow,
               uint32_t kcol, uint32_t inrow, uint32_t incol, uint32_t startrow,
               uint32_t startcol) {
  BuckyballInstruction inst = build_instruction(IM2COL_FUNC7);
  InstructionBuilder builder = create_builder(&inst, IM2COL_FUNC7);

  builder.set.rs1(&builder, "op_spaddr", op1_addr);
  builder.set.rs1(&builder, "wr_spaddr", wr_addr);
  builder.set.rs2(&builder, "krow", krow);
  builder.set.rs2(&builder, "kcol", kcol);
  builder.set.rs2(&builder, "inrow", inrow);
  builder.set.rs2(&builder, "incol", incol);
  builder.set.rs2(&builder, "startrow", startrow);
  builder.set.rs2(&builder, "startcol", startcol);

  execute_builder(builder);
}
