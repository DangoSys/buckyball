#include "isa.h"

// MVOUT指令配置
extern "C" const InstructionConfig mvout_config = {
    .rs1_fields = (BitFieldConfig[]){{"base_dram_addr", 0, 31}, {NULL, 0, 0}},
    .rs2_fields = (BitFieldConfig[]){
        {"base_sp_addr", 0, 13}, {"iter", 14, 35}, {NULL, 0, 0}}};

// MVOUT指令高级API实现
void bb_mvout(uint64_t mem_addr, uint32_t sp_addr, uint32_t iter) {
  BuckyballInstruction inst = build_instruction(MVOUT_FUNC7);
  InstructionBuilder builder = create_builder(&inst, MVOUT_FUNC7);

  builder.set.rs1(&builder, "base_dram_addr", (uint32_t)mem_addr);
  builder.set.rs2(&builder, "base_sp_addr", sp_addr);
  builder.set.rs2(&builder, "iter", iter);

  execute_builder(builder);
}
