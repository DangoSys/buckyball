#include "isa.h"

// TRANSPOSE指令配置
const InstructionConfig transpose_config = {
    .rs1_fields = (BitFieldConfig[]){{"op_spaddr", 0, 13},
                                     {"wr_spaddr", 14, 27},
                                     {NULL, 0, 0}},
    .rs2_fields = (BitFieldConfig[]){{"iter", 14, 35}, {NULL, 0, 0}}};

// TRANSPOSE指令执行函数
#ifndef __x86_64__
static void execute_transpose_impl(uint32_t rs1_val, uint32_t rs2_val) {
  asm volatile(".insn r " STR(CUSTOM_3) ", 0x3, 34, x0, %0, %1"
               : : "r"(rs1_val), "r"(rs2_val) : "memory");
}
#else
static void execute_transpose_impl(uint32_t rs1_val, uint32_t rs2_val) {
  // x86平台下不执行RISC-V指令
}
#endif

// 注册TRANSPOSE指令
void register_transpose_instruction(void) {
  register_instruction(TRANSPOSE_FUNC7, execute_transpose_impl);
}

// TRANSPOSE指令高级API实现
void bb_transpose(uint32_t op1_addr, uint32_t wr_addr, uint32_t iter) {
  BuckyballInstruction inst = build_instruction(TRANSPOSE_FUNC7);
  InstructionBuilder builder = create_builder(&inst, TRANSPOSE_FUNC7);

  builder.set.rs1(&builder, "op_spaddr", op1_addr);
  builder.set.rs1(&builder, "wr_spaddr", wr_addr);
  builder.set.rs2(&builder, "iter", iter);

  execute_builder(builder);
}
