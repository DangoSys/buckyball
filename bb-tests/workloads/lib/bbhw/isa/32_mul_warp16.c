#include "isa.h"

// MUL_WARP16指令配置
const InstructionConfig mul_warp16_config = {
    .rs1_fields = (BitFieldConfig[]){{"op1_spaddr", 0, 13},
                                     {"op2_spaddr", 14, 27},
                                     {NULL, 0, 0}},
    .rs2_fields = (BitFieldConfig[]){
        {"wr_spaddr", 0, 13}, {"iter", 14, 35}, {NULL, 0, 0}}};

// MUL_WARP16指令执行函数
#ifndef __x86_64__
static void execute_mul_warp16_impl(uint32_t rs1_val, uint32_t rs2_val) {
  asm volatile(".insn r " STR(CUSTOM_3) ", 0x3, 32, x0, %0, %1"
               : : "r"(rs1_val), "r"(rs2_val) : "memory");
}
#else
static void execute_mul_warp16_impl(uint32_t rs1_val, uint32_t rs2_val) {
  // x86平台下不执行RISC-V指令
}
#endif

// 注册MUL_WARP16指令
void register_mul_warp16_instruction(void) {
  register_instruction(MUL_WARP16_FUNC7, execute_mul_warp16_impl);
}

// MUL_WARP16指令高级API实现
void bb_mul_warp16(uint32_t op1_addr, uint32_t op2_addr, uint32_t wr_addr,
                   uint32_t iter) {
  BuckyballInstruction inst = build_instruction(MUL_WARP16_FUNC7);
  InstructionBuilder builder = create_builder(&inst, MUL_WARP16_FUNC7);

  builder.set.rs1(&builder, "op1_spaddr", op1_addr);
  builder.set.rs1(&builder, "op2_spaddr", op2_addr);
  builder.set.rs2(&builder, "wr_spaddr", wr_addr);
  builder.set.rs2(&builder, "iter", iter);

  execute_builder(builder);
}
