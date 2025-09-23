#include "isa.h"
#include <string.h>

// 通用字段操作函数
uint32_t get_bbinst_field(uint64_t value, const char *field_name,
                          const BitFieldConfig *config) {
  for (int i = 0; config[i].name != NULL; i++) {
    if (strcmp(config[i].name, field_name) == 0) {
      uint32_t bit_width = config[i].end_bit - config[i].start_bit + 1;
      uint64_t mask = ((1ULL << bit_width) - 1);
      return (value >> config[i].start_bit) & mask;
    }
  }
  return 0; // 字段未找到
}

void set_bbinst_field(uint64_t *value, const char *field_name,
                      uint32_t field_value, const BitFieldConfig *config) {
  for (int i = 0; config[i].name != NULL; i++) {
    if (strcmp(config[i].name, field_name) == 0) {
      uint32_t bit_width = config[i].end_bit - config[i].start_bit + 1;
      uint64_t mask = ((1ULL << bit_width) - 1);
      // 清除原有值
      *value &= ~(mask << config[i].start_bit);
      // 设置新值
      *value |= ((uint64_t)(field_value & mask) << config[i].start_bit);
      return;
    }
  }
}

// 指令构建和操作函数
BuckyballInstruction build_instruction(InstructionType type) {
  BuckyballInstruction inst = {0, 0};
  return inst;
}

// 内部设置函数实现
InstructionBuilder set_rs1_internal(void *builder_ptr, const char *field_name,
                                    uint32_t value) {
  InstructionBuilder *builder = (InstructionBuilder *)builder_ptr;
  const InstructionConfig *cfg = config(builder->type);
  set_bbinst_field(&builder->inst->rs1, field_name, value, cfg->rs1_fields);
  return *builder;
}

InstructionBuilder set_rs2_internal(void *builder_ptr, const char *field_name,
                                    uint32_t value) {
  InstructionBuilder *builder = (InstructionBuilder *)builder_ptr;
  const InstructionConfig *cfg = config(builder->type);
  set_bbinst_field(&builder->inst->rs2, field_name, value, cfg->rs2_fields);
  return *builder;
}

InstructionBuilder create_builder(BuckyballInstruction *inst,
                                  InstructionType type) {
  InstructionBuilder builder = {.inst = inst,
                                .type = type,
                                .set = {.builder_ptr = NULL, // 将在使用时设置
                                        .rs1 = set_rs1_internal,
                                        .rs2 = set_rs2_internal}};
  // 设置自引用指针
  builder.set.builder_ptr = &builder;
  return builder;
}

// 外部配置声明 - 纯C实现
extern const InstructionConfig mvin_config;
extern const InstructionConfig mvout_config;
extern const InstructionConfig mul_warp16_config;
extern const InstructionConfig bbfp_mul_config;
extern const InstructionConfig matmul_ws_config;
extern const InstructionConfig im2col_config;
extern const InstructionConfig transpose_config;

// 通过func7获取指令配置
const InstructionConfig *config(InstructionType func7) {
  switch (func7) {
  case MVIN_FUNC7:
    return &mvin_config;
  case MVOUT_FUNC7:
    return &mvout_config;
  case MUL_WARP16_FUNC7:
    return &mul_warp16_config;
  case BBFP_MUL_FUNC7:
    return &bbfp_mul_config;
  case MATMUL_WS_FUNC7:
    return &matmul_ws_config;
  case IM2COL_FUNC7:
    return &im2col_config;
  case TRANSPOSE_FUNC7:
    return &transpose_config;
  case FENCE_FUNC7:
    return NULL; // FENCE指令没有参数，不需要配置
  case FLUSH_FUNC7:
    return NULL; // FLUSH指令没有参数，不需要配置
  default:
    return NULL;
  }
}

void execute_builder(const InstructionBuilder builder) {
#ifdef __x86_64__
  // x86平台下不执行RISC-V指令
#else
  uint32_t rs1_val = (uint32_t)builder.inst->rs1;
  uint32_t rs2_val = (uint32_t)builder.inst->rs2;

  asm volatile(".insn r " STR(CUSTOM_3) ", " STR(0x3) ", %2, x0, %0, %1"
               :
               : "r"(rs1_val), "r"(rs2_val), "i"(builder.type));
  // 使用 switch 确保每个分支中的 func7 是编译时常量
  // switch (builder.type) {
  // case MVIN_FUNC7:
  //   asm volatile(".insn r " STR(CUSTOM_3) ", " STR(0x3) ", %2, x0, %0, %1"
  //                :
  //                : "r"(rs1_val), "r"(rs2_val), "i"(MVIN_FUNC7));
  //   break;
  // case MVOUT_FUNC7:
  //   asm volatile(".insn r " STR(CUSTOM_3) ", " STR(0x3) ", %2, x0, %0, %1"
  //                :
  //                : "r"(rs1_val), "r"(rs2_val), "i"(MVOUT_FUNC7));
  //   break;
  // case FENCE_FUNC7:
  //   asm volatile(".insn r " STR(CUSTOM_3) ", " STR(0x3) ", %2, x0, %0, %1"
  //                :
  //                : "r"(rs1_val), "r"(rs2_val), "i"(FENCE_FUNC7));
  //   break;
  // case MUL_WARP16_FUNC7:
  //   asm volatile(".insn r " STR(CUSTOM_3) ", " STR(0x3) ", %2, x0, %0, %1"
  //                :
  //                : "r"(rs1_val), "r"(rs2_val), "i"(MUL_WARP16_FUNC7));
  //   break;
  // case BBFP_MUL_FUNC7:
  //   asm volatile(".insn r " STR(CUSTOM_3) ", " STR(0x3) ", %2, x0, %0, %1"
  //                :
  //                : "r"(rs1_val), "r"(rs2_val), "i"(BBFP_MUL_FUNC7));
  //   break;
  // case MATMUL_WS_FUNC7:
  //   asm volatile(".insn r " STR(CUSTOM_3) ", " STR(0x3) ", %2, x0, %0, %1"
  //                :
  //                : "r"(rs1_val), "r"(rs2_val), "i"(MATMUL_WS_FUNC7));
  //   break;
  // case IM2COL_FUNC7:
  //   asm volatile(".insn r " STR(CUSTOM_3) ", " STR(0x3) ", %2, x0, %0, %1"
  //                :
  //                : "r"(rs1_val), "r"(rs2_val), "i"(IM2COL_FUNC7));
  //   break;
  // case TRANSPOSE_FUNC7:
  //   asm volatile(".insn r " STR(CUSTOM_3) ", " STR(0x3) ", %2, x0, %0, %1"
  //                :
  //                : "r"(rs1_val), "r"(rs2_val), "i"(TRANSPOSE_FUNC7));
  //   break;
  // case FLUSH_FUNC7:
  //   asm volatile(".insn r " STR(CUSTOM_3) ", " STR(0x3) ", %2, x0, %0, %1"
  //                :
  //                : "r"(rs1_val), "r"(rs2_val), "i"(FLUSH_FUNC7));
  //   break;
  // default:
  //   // 未知指令类型，不执行
  //   break;
  // }
#endif
}
