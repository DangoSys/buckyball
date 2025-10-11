#ifndef BUCKYBALL_ISA_H
#define BUCKYBALL_ISA_H

#include <stddef.h>
#include <stdint.h>

/* Pure C implementation - no C++ linkage needed */

// Data type for matrix elements
typedef int8_t elem_t;
typedef int32_t result_t;

// Custom instruction opcodes
#define CUSTOM_3 0x7b
// String macros (from xcustom.h)
#define STR1(x) #x
#ifndef STR
#define STR(x) STR1(x)
#endif

// 通用字段编码宏
#define ENCODE_FIELD(value, start_bit, width)                                  \
  (((value) & ((1ULL << (width)) - 1)) << (start_bit))

// 位字段配置结构
typedef struct {
  const char *name;   // 字段名称 (NULL表示数组结束)
  uint32_t start_bit; // 起始位
  uint32_t end_bit;   // 结束位(包含)
} BitFieldConfig;

// 指令类型枚举 - 直接使用func7值
typedef enum {
  MVIN_FUNC7 = 24,       // 0x18 - Move in function code
  MVOUT_FUNC7 = 25,      // 0x19 - Move out function code
  FENCE_FUNC7 = 31,      // 0x1F - Fence function code
  MUL_WARP16_FUNC7 = 32, // 0x20 - Matrix multiply function code
  IM2COL_FUNC7 = 33,     // 0x21 - Matrix im2col function code
  TRANSPOSE_FUNC7 = 34,  // 0x22 - Matrix transpose function code
  GELU_FUNC7 = 35,       // 0x23 - GELU activation function code
  LAYERNORM_FUNC7 = 36,  // 0x24 - LayerNorm function code
  SOFTMAX_FUNC7 = 37,    // 0x25 - Softmax function code
  FLUSH_FUNC7 = 7,       // 0x07 - Flush function code
  BBFP_MUL_FUNC7 = 26,   // 0x1A - BBFP matrix multiply function code
  MATMUL_WS_FUNC7 = 27   // 0x1B - Matrix multiply with warp16 function code
} InstructionType;

// 指令配置结构 (for simulator)
typedef struct {
  const BitFieldConfig *rs1_fields; // rs1寄存器的字段配置 (以NULL name结尾)
  const BitFieldConfig *rs2_fields; // rs2寄存器的字段配置 (以NULL name结尾)
} InstructionConfig;

// 通用字段获取函数 (for simulator)
uint32_t get_bbinst_field(uint64_t value, const char *field_name,
                          const BitFieldConfig *config);
void set_bbinst_field(uint64_t *value, const char *field_name,
                      uint32_t field_value, const BitFieldConfig *config);

// 高级别API (for CTest)

void bb_mvin(uint64_t mem_addr, uint32_t sp_addr, uint32_t iter,
             uint32_t col_stride);
void bb_mvout(uint64_t mem_addr, uint32_t sp_addr, uint32_t iter);
void bb_fence(void);
void bb_mul_warp16(uint32_t op1_addr, uint32_t op2_addr, uint32_t wr_addr,
                   uint32_t iter);
void bb_bbfp_mul(uint32_t op1_addr, uint32_t op2_addr, uint32_t wr_addr,
                 uint32_t iter);
void bb_matmul_ws(uint32_t op1_addr, uint32_t op2_addr, uint32_t wr_addr,
                  uint32_t iter);
void bb_im2col(uint32_t op1_addr, uint32_t wr_addr, uint32_t krow,
               uint32_t kcol, uint32_t inrow, uint32_t incol, uint32_t startrow,
               uint32_t startcol);
void bb_transpose(uint32_t op1_addr, uint32_t wr_addr, uint32_t iter);
void bb_gelu(uint32_t op1_addr, uint32_t wr_addr, uint32_t iter);
void bb_layernorm_simple(uint32_t op1_bank, uint32_t op1_addr, uint32_t wr_bank,
                         uint32_t wr_addr, uint32_t iter, uint32_t is_acc,
                         uint32_t norm_dim);
void bb_layernorm(uint32_t op1_bank, uint32_t op1_addr, uint32_t wr_bank,
                  uint32_t wr_addr, uint32_t iter, uint32_t is_acc,
                  uint32_t norm_dim, uint32_t gamma_addr, uint32_t beta_addr,
                  uint32_t param_bank, uint32_t use_affine);
void bb_softmax(uint32_t op1_bank, uint32_t op1_addr, uint32_t wr_bank,
                uint32_t wr_addr, uint32_t iter, uint32_t is_acc,
                uint32_t dim_len, uint32_t batch, uint32_t log_mode);
void bb_flush(void);

// 通过func7获取指令配置
const InstructionConfig *config(InstructionType func7);

/* End of pure C header */

#endif // BUCKYBALL_ISA_H
