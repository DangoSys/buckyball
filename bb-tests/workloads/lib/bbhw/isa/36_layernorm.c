#include "isa.h"

// =========================== for simulator ===========================
const InstructionConfig layernorm_config = {
    .rs1_fields = (BitFieldConfig[]){{"op1_bank", 0, 2},
                                     {"op1_spaddr", 3, 14},
                                     {"wr_bank", 15, 17},
                                     {"wr_spaddr", 18, 29},
                                     {NULL, 0, 0}},
    .rs2_fields = (BitFieldConfig[]){{"iter", 0, 9},
                                     {"is_acc", 10, 10},
                                     {"norm_dim", 11, 22},
                                     {"gamma_addr", 23, 34},
                                     {"beta_addr", 35, 46},
                                     {"param_bank", 47, 49},
                                     {"use_affine", 50, 50},
                                     {NULL, 0, 0}}};

// =========================== for CTest ===========================
// RS1: [op1_bank(3) | op1_addr(12) | wr_bank(3) | wr_addr(12)]
#define LAYERNORM_ENCODE_RS1(op1_bank, op1_addr, wr_bank, wr_addr)             \
  (ENCODE_FIELD(op1_bank, 0, 3) | ENCODE_FIELD(op1_addr, 3, 12) |              \
   ENCODE_FIELD(wr_bank, 15, 3) | ENCODE_FIELD(wr_addr, 18, 12))

// RS2: [iter(10) | is_acc(1) | norm_dim(12) | gamma_addr(12) |
//       beta_addr(12) | param_bank(2) | use_affine(1)]
#define LAYERNORM_ENCODE_RS2(iter, is_acc, norm_dim, gamma_addr, beta_addr,    \
                             param_bank, use_affine)                           \
  (ENCODE_FIELD(iter, 0, 10) | ENCODE_FIELD(is_acc, 10, 1) |                   \
   ENCODE_FIELD(norm_dim, 11, 12) | ENCODE_FIELD(gamma_addr, 23, 12) |         \
   ENCODE_FIELD(beta_addr, 35, 12) | ENCODE_FIELD(param_bank, 47, 2) |         \
   ENCODE_FIELD(use_affine, 49, 1))

// LayerNorm指令低级实现
#ifndef __x86_64__
#define LAYERNORM_RAW(rs1, rs2)                                                \
  asm volatile(".insn r " STR(CUSTOM_3) ", 0x3, 36, x0, %0, %1"                \
               :                                                               \
               : "r"(rs1), "r"(rs2)                                            \
               : "memory")
#else
#define LAYERNORM_RAW(rs1, rs2) /* x86平台下不执行RISC-V指令 */
#endif

// LayerNorm指令高级API实现 - 简化版（无仿射变换）
void bb_layernorm_simple(uint32_t op1_bank, uint32_t op1_addr, uint32_t wr_bank,
                         uint32_t wr_addr, uint32_t iter, uint32_t is_acc,
                         uint32_t norm_dim) {
  uint64_t rs1_val = LAYERNORM_ENCODE_RS1(op1_bank, op1_addr, wr_bank, wr_addr);
  uint64_t rs2_val = LAYERNORM_ENCODE_RS2(iter, is_acc, norm_dim, 0, 0, 0, 0);
  LAYERNORM_RAW(rs1_val, rs2_val);
}

// LayerNorm指令高级API实现 - 完整版（带仿射变换）
void bb_layernorm(uint32_t op1_bank, uint32_t op1_addr, uint32_t wr_bank,
                  uint32_t wr_addr, uint32_t iter, uint32_t is_acc,
                  uint32_t norm_dim, uint32_t gamma_addr, uint32_t beta_addr,
                  uint32_t param_bank, uint32_t use_affine) {
  uint64_t rs1_val = LAYERNORM_ENCODE_RS1(op1_bank, op1_addr, wr_bank, wr_addr);
  uint64_t rs2_val = LAYERNORM_ENCODE_RS2(iter, is_acc, norm_dim, gamma_addr,
                                          beta_addr, param_bank, use_affine);
  LAYERNORM_RAW(rs1_val, rs2_val);
}
