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
  // Field not found
  return 0;
}

void set_bbinst_field(uint64_t *value, const char *field_name,
                      uint32_t field_value, const BitFieldConfig *config) {
  for (int i = 0; config[i].name != NULL; i++) {
    if (strcmp(config[i].name, field_name) == 0) {
      uint32_t bit_width = config[i].end_bit - config[i].start_bit + 1;
      uint64_t mask = ((1ULL << bit_width) - 1);
      // Clear original value
      *value &= ~(mask << config[i].start_bit);
      // Set new value
      *value |= ((uint64_t)(field_value & mask) << config[i].start_bit);
      return;
    }
  }
}

// External configuration declarations - defined in individual instruction files
extern const InstructionConfig mvin_config;
extern const InstructionConfig mvout_config;
extern const InstructionConfig mul_warp16_config;
extern const InstructionConfig bbfp_mul_config;
extern const InstructionConfig matmul_ws_config;
extern const InstructionConfig im2col_config;
extern const InstructionConfig transpose_config;
extern const InstructionConfig relu_config;
extern const InstructionConfig bbus_config_config;
extern const InstructionConfig nnlut_config;
extern const InstructionConfig snn_config;
extern const InstructionConfig abft_systolic_config;
extern const InstructionConfig conv_config;
extern const InstructionConfig cim_config;

// Get instruction configuration by func7
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
  case NNLUT_FUNC7:
    return &nnlut_config;
  case SNN_FUNC7:
    return &snn_config;
  case ABFT_SYSTOLIC_FUNC7:
    return &abft_systolic_config;
  case CONV_FUNC7:
    return &conv_config;
  case CIM_FUNC7:
    return &cim_config;
  case FENCE_FUNC7:
    // FENCE instruction has no parameters, no configuration needed
    return NULL;
  case FLUSH_FUNC7:
    // FLUSH instruction has no parameters, no configuration needed
    return NULL;
  default:
    return NULL;
  }
}
