#include "isa.h"
#include <string.h>

// 指令注册表
#define MAX_INSTRUCTIONS 16
static InstructionRegistry instruction_registry[MAX_INSTRUCTIONS];
static int registry_count = 0;

// 注册所有指令的便捷函数
void register_all_instructions(void) {
  static int initialized = 0;
  if (initialized) return;
  
  // 调用各个指令文件的注册函数
  register_mvin_instruction();
  register_mvout_instruction();
  register_fence_instruction();
  register_mul_warp16_instruction();
  register_im2col_instruction();
  register_transpose_instruction();
  register_flush_instruction();
  register_bbfp_mul_instruction();
  register_matmul_ws_instruction();
  
  initialized = 1;
}

// 注册指令执行函数
void register_instruction(InstructionType type, InstructionExecutor executor) {
  if (registry_count < MAX_INSTRUCTIONS) {
    instruction_registry[registry_count].type = type;
    instruction_registry[registry_count].executor = executor;
    registry_count++;
  }
}

// 执行指定类型的指令
void execute_instruction(InstructionType type, uint32_t rs1_val, uint32_t rs2_val) {
  register_all_instructions();
  
  for (int i = 0; i < registry_count; i++) {
    if (instruction_registry[i].type == type) {
      instruction_registry[i].executor(rs1_val, rs2_val);
      return;
    }
  }
  // 未找到对应的指令执行函数
}

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
  uint32_t rs1_val = (uint32_t)builder.inst->rs1;
  uint32_t rs2_val = (uint32_t)builder.inst->rs2;
  execute_instruction(builder.type, rs1_val, rs2_val);
}
