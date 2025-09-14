#include "isa.h"

// TRANSPOSE指令配置
extern "C" const InstructionConfig transpose_config = {
    .rs1_fields = (BitFieldConfig[]){{"op_spaddr", 0, 13},
                                     {"wr_spaddr", 14, 27},
                                     {NULL, 0, 0}},
    .rs2_fields = (BitFieldConfig[]){{"iter", 14, 35}, {NULL, 0, 0}}};

// TRANSPOSE指令高级API实现
void bb_transpose(uint32_t op1_addr, uint32_t wr_addr, uint32_t iter) {
  BuckyballInstruction inst = build_instruction(TRANSPOSE_FUNC7);
  InstructionBuilder builder = create_builder(&inst, TRANSPOSE_FUNC7);

  builder.set.rs1(&builder, "op_spaddr", op1_addr);
  builder.set.rs1(&builder, "wr_spaddr", wr_addr);
  builder.set.rs2(&builder, "iter", iter);

  execute_builder(builder);
}
