#ifndef BUCKYBALL_ISA_H
#define BUCKYBALL_ISA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

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
  FLUSH_FUNC7 = 7,       // 0x07 - Flush function code
  BBFP_MUL_FUNC7 = 26,   // 0x1A - BBFP matrix multiply function code
  MATMUL_WS_FUNC7 = 27   // 0x1B - Matrix multiply with warp16 function code
} InstructionType;

// 通用指令结构
typedef struct {
  uint64_t rs1; // 第一个寄存器
  uint64_t rs2; // 第二个寄存器
} BuckyballInstruction;

// 指令配置结构
typedef struct {
  const BitFieldConfig *rs1_fields; // rs1寄存器的字段配置 (以NULL name结尾)
  const BitFieldConfig *rs2_fields; // rs2寄存器的字段配置 (以NULL name结尾)
} InstructionConfig;

// 外部配置数组声明 - 通过func7索引
extern const InstructionConfig *instruction_configs;

// 通用字段设置/获取函数
uint32_t get_bbinst_field(uint64_t value, const char *field_name,
                          const BitFieldConfig *config);
void set_bbinst_field(uint64_t *value, const char *field_name,
                      uint32_t field_value, const BitFieldConfig *config);

// 前向声明
typedef struct InstructionBuilder InstructionBuilder;

// 寄存器设置器结构
typedef struct {
  void *builder_ptr; // 指向InstructionBuilder的指针
  InstructionBuilder (*rs1)(void *builder, const char *field_name,
                            uint32_t value);
  InstructionBuilder (*rs2)(void *builder, const char *field_name,
                            uint32_t value);
} RegisterSetter;

// 指令构建器结构
struct InstructionBuilder {
  BuckyballInstruction *inst;
  InstructionType type;
  RegisterSetter set; // 内嵌的设置器
};

// 内部函数声明
InstructionBuilder set_rs1_internal(void *builder, const char *field_name,
                                    uint32_t value);
InstructionBuilder set_rs2_internal(void *builder, const char *field_name,
                                    uint32_t value);

// 指令构建和执行函数
BuckyballInstruction build_instruction(InstructionType type);
InstructionBuilder create_builder(BuckyballInstruction *inst,
                                  InstructionType type);
void execute_builder(const InstructionBuilder builder);

// 高级别API
void generate_inst(uint32_t func7, uint32_t rs1_val, uint32_t rs2_val);

void bb_mvin(uint64_t mem_addr, uint32_t sp_addr, uint32_t iter);
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
void bb_flush(void);

// 通过func7获取指令配置
const InstructionConfig *config(InstructionType func7);

#ifdef __cplusplus
}
#endif

#endif // BUCKYBALL_ISA_H
