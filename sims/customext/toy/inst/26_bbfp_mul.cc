#include "common.h"
#include "inst.h"
#include "toy.h"

// Matrix multiplication using bbfp pattern
void toy_t::bbfp_mul(reg_t rs1, reg_t rs2) {
  bbfp_mul_rs1_t rs1_fields(rs1);
  bbfp_mul_rs2_t rs2_fields(rs2);

  auto const op1_spaddr = rs1_fields.op1_spaddr();
  auto const op2_spaddr = rs1_fields.op2_spaddr();
  auto const wr_spaddr = rs2_fields.wr_spaddr();
  auto const iter = rs2_fields.iter();

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
