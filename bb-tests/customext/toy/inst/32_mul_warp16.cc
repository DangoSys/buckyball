#include "common.h"
#include "toy.h"
#include <bbhw/isa/isa.h>

// Matrix multiplication using warp16 pattern
void toy_t::mul_warp16(reg_t rs1, reg_t rs2) {
  // 使用库中的字段提取函数
  const InstructionConfig *cfg = config(MUL_WARP16_FUNC7);

  auto const op1_spaddr = get_bbinst_field(rs1, "op1_spaddr", cfg->rs1_fields);
  auto const op2_spaddr = get_bbinst_field(rs1, "op2_spaddr", cfg->rs1_fields);
  auto const wr_accaddr = get_bbinst_field(rs2, "wr_spaddr", cfg->rs2_fields);
  auto const iter = get_bbinst_field(rs2, "iter", cfg->rs2_fields);

  dprintf(p, "TOY: mul_warp16 - rs1=0x%08lx, rs2=0x%08lx\n", rs1, rs2);
  dprintf(p,
          "TOY: mul_warp16 - op1_spaddr=0x%08lx, op2_spaddr=0x%08lx, "
          "wr_accaddr=0x%08lx, iter=0x%02lx\n",
          op1_spaddr, op2_spaddr, wr_accaddr, iter);

  uint32_t op1_bank = spad_get_bank(op1_spaddr);
  uint32_t op1_offset = spad_get_offset(op1_spaddr);
  uint32_t op2_bank = spad_get_bank(op2_spaddr);
  uint32_t op2_offset = spad_get_offset(op2_spaddr);
  uint32_t acc_bank = spad_get_bank(wr_accaddr);
  uint32_t acc_offset = spad_get_offset(wr_accaddr);

  // iter在外层：每次迭代做标量向量乘并累加
  for (size_t k = 0; k < iter; ++k) {
    for (size_t i = 0; i < DIM; ++i) {
      elem_t a = read_from_bank<elem_t>(toy_state, op1_bank, op1_offset + k, i);

      for (size_t j = 0; j < DIM; ++j) {
        elem_t b =
            read_from_bank<elem_t>(toy_state, op2_bank, op2_offset + k, j);
        acc_t prev_result =
            read_from_bank<acc_t>(toy_state, acc_bank, acc_offset + i, j);
        acc_t new_result = prev_result + (acc_t)a * (acc_t)b;
        write_to_bank<acc_t>(toy_state, acc_bank, acc_offset + i, j,
                             new_result);
      }
    }
  }
}
