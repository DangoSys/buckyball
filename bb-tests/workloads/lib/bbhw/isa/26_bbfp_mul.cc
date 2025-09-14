#include "isa.h"

// BBFP_MUL指令配置
extern "C" const InstructionConfig bbfp_mul_config = {
    .rs1_fields = (BitFieldConfig[]){{"op1_spaddr", 0, 13},
                                     {"op2_spaddr", 14, 27},
                                     {NULL, 0, 0}},
    .rs2_fields = (BitFieldConfig[]){
        {"wr_spaddr", 0, 13}, {"iter", 14, 35}, {NULL, 0, 0}}};

// BBFP_MUL指令高级API实现
void bb_bbfp_mul(uint32_t op1_addr, uint32_t op2_addr, uint32_t wr_addr,
                 uint32_t iter) {
  BuckyballInstruction inst = build_instruction(BBFP_MUL_FUNC7);
  InstructionBuilder builder = create_builder(&inst, BBFP_MUL_FUNC7);

  builder.set.rs1(&builder, "op1_spaddr", op1_addr);
  builder.set.rs1(&builder, "op2_spaddr", op2_addr);
  builder.set.rs2(&builder, "wr_spaddr", wr_addr);
  builder.set.rs2(&builder, "iter", iter);

  execute_builder(builder);
}
