#include "isa.h"

// MVIN指令配置
const InstructionConfig mvin_config = {
    .rs1_fields = (BitFieldConfig[]){{"base_dram_addr", 0, 31}, {NULL, 0, 0}},
    .rs2_fields = (BitFieldConfig[]){
        {"base_sp_addr", 0, 13}, {"iter", 14, 35}, {NULL, 0, 0}}};

// MVIN指令执行函数
#ifndef __x86_64__
static void execute_mvin_impl(uint32_t rs1_val, uint32_t rs2_val) {
  asm volatile(".insn r " STR(CUSTOM_3) ", 0x3, 24, x0, %0, %1"
               : : "r"(rs1_val), "r"(rs2_val) : "memory");
}
#else
static void execute_mvin_impl(uint32_t rs1_val, uint32_t rs2_val) {
  // x86平台下不执行RISC-V指令
}
#endif

// 注册MVIN指令
void register_mvin_instruction(void) {
  register_instruction(MVIN_FUNC7, execute_mvin_impl);
}

// MVIN指令高级API实现
void bb_mvin(uint64_t mem_addr, uint32_t sp_addr, uint32_t iter) {
  BuckyballInstruction inst = build_instruction(MVIN_FUNC7);
  InstructionBuilder builder = create_builder(&inst, MVIN_FUNC7);

  builder.set.rs1(&builder, "base_dram_addr", (uint32_t)mem_addr);
  builder.set.rs2(&builder, "base_sp_addr", sp_addr);
  builder.set.rs2(&builder, "iter", iter);

  execute_builder(builder);
}
