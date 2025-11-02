#include "common.h"
#include "toy.h"
#include <bbhw/isa/isa.h>

// Matrix multiplication using bbfp pattern
void toy_t::bbfp_mul(reg_t rs1, reg_t rs2) {
  // Use field extraction functions from library
  const InstructionConfig *cfg = config(BBFP_MUL_FUNC7);

  auto const op1_spaddr = get_bbinst_field(rs1, "op1_spaddr", cfg->rs1_fields);
  auto const op2_spaddr = get_bbinst_field(rs1, "op2_spaddr", cfg->rs1_fields);
  auto const wr_spaddr = get_bbinst_field(rs2, "wr_spaddr", cfg->rs2_fields);
  auto const iter = get_bbinst_field(rs2, "iter", cfg->rs2_fields);

  dprintf(p, "TOY: bbfp_mul - rs1=0x%08lx, rs2=0x%08lx\n", rs1, rs2);
  dprintf(p,
          "TOY: bbfp_mul - op1_spaddr=0x%08lx, op2_spaddr=0x%08lx, "
          "wr_spaddr=0x%08lx, iter=0x%02lx\n",
          op1_spaddr, op2_spaddr, wr_spaddr, iter);

  // Perform matrix multiplication for specified iterations
  for (size_t i = 0; i < iter; ++i) {
    // For each iteration, compute one row of result matrix
    const size_t result_row = wr_spaddr + i;
    const size_t op1_row = op1_spaddr + i;
    const size_t op2_row = op2_spaddr + i;
    uint32_t op1_bank = spad_get_bank(op1_row);
    uint32_t op1_offset = spad_get_offset(op1_row);
    uint32_t op2_bank = spad_get_bank(op2_row);
    uint32_t op2_offset = spad_get_offset(op2_row);
    uint32_t result_bank = spad_get_bank(result_row);
    uint32_t result_offset = spad_get_offset(result_row);

    elem_t share_exp_a = toy_state.banks.at(op1_bank).at(op1_offset).at(0);
    elem_t share_exp_b = toy_state.banks.at(op2_bank).at(op2_offset).at(0);

    elem_t sum_exp = share_exp_a + share_exp_b;
    toy_state.banks.at(result_bank).at(result_offset).at(0) = sum_exp;

    for (size_t col = 1; col < DIM; ++col) {
      elem_t sum = 0;
      for (size_t k = 1; k < DIM; ++k) {
        elem_t a = toy_state.banks.at(op1_bank).at(op1_offset).at(k);
        elem_t b = toy_state.banks.at(op2_bank).at(op2_offset).at(k);
        sum += a * b;
      }
      toy_state.banks.at(result_bank).at(result_offset).at(col) = sum;
    }
  }
}
