#include "isa.h"

// MUL_WARP16指令配置
extern "C" const InstructionConfig mul_warp16_config = {
    .rs1_fields = (BitFieldConfig[]){{"op1_spaddr", 0, 13},
                                     {"op2_spaddr", 14, 27},
                                     {NULL, 0, 0}},
    .rs2_fields = (BitFieldConfig[]){
        {"wr_spaddr", 0, 13}, {"iter", 14, 35}, {NULL, 0, 0}}};

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
