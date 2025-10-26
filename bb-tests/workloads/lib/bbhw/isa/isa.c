#include "isa.h"
#include <string.h>

// =========================== for simulator ===========================
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

// 外部配置声明 - 各个指令文件中定义
extern const InstructionConfig mvin_config;
extern const InstructionConfig mvout_config;
extern const InstructionConfig mul_warp16_config;
extern const InstructionConfig bbfp_mul_config;
extern const InstructionConfig matmul_ws_config;
extern const InstructionConfig im2col_config;
extern const InstructionConfig transpose_config;
extern const InstructionConfig relu_config;
extern const InstructionConfig bbus_config_config;

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
  case RELU_FUNC7:
    return &relu_config;
  case BBUS_CONFIG_FUNC7:
    return &bbus_config_config;
  case FENCE_FUNC7:
    return NULL; // FENCE指令没有参数，不需要配置
  case FLUSH_FUNC7:
    return NULL; // FLUSH指令没有参数，不需要配置
  default:
    return NULL;
  }
}
