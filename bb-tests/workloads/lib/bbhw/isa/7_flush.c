#include "isa.h"

// FLUSH指令执行函数
#ifndef __x86_64__
static void execute_flush_impl(uint32_t rs1_val, uint32_t rs2_val) {
  asm volatile(".insn r " STR(CUSTOM_3) ", 0x3, 7, x0, %0, %1"
               : : "r"(rs1_val), "r"(rs2_val) : "memory");
}
#else
static void execute_flush_impl(uint32_t rs1_val, uint32_t rs2_val) {
  // x86平台下不执行RISC-V指令
}
#endif

// 注册FLUSH指令
void register_flush_instruction(void) {
  register_instruction(FLUSH_FUNC7, execute_flush_impl);
}

// FLUSH指令实现
void bb_flush(void) {
  BuckyballInstruction inst = {0, 0};
  InstructionBuilder builder = create_builder(&inst, FLUSH_FUNC7);
  execute_builder(builder);
}
