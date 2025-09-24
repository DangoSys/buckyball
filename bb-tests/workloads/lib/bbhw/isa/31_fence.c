#include "isa.h"

// FENCE指令执行函数
#ifndef __x86_64__
static void execute_fence_impl(uint32_t rs1_val, uint32_t rs2_val) {
  asm volatile(".insn r " STR(CUSTOM_3) ", 0x3, 31, x0, %0, %1"
               : : "r"(rs1_val), "r"(rs2_val) : "memory");
}
#else
static void execute_fence_impl(uint32_t rs1_val, uint32_t rs2_val) {
  // x86平台下不执行RISC-V指令
}
#endif

// 注册FENCE指令
void register_fence_instruction(void) {
  register_instruction(FENCE_FUNC7, execute_fence_impl);
}

// FENCE指令实现
void bb_fence(void) {
  BuckyballInstruction inst = {0, 0};
  InstructionBuilder builder = create_builder(&inst, FENCE_FUNC7);
  execute_builder(builder);
}
