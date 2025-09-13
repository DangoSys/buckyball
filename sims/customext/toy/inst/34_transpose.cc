#include "common.h"
#include "inst.h"
#include "toy.h"

#include <cassert>

void toy_t::transpose(reg_t rs1, reg_t rs2) {
  transpose_rs1_t rs1_fields(rs1);
  transpose_rs2_t rs2_fields(rs2);

  auto const op_spaddr = rs1_fields.op_spaddr();
  auto const wr_spaddr = rs1_fields.wr_spaddr();
  auto const iter = rs2_fields.iter();

  dprintf(p, "TOY: transpose - rs1=%lx, rs2=%lx\n", rs1, rs2);
  dprintf(p, "TOY: transpose - op_spaddr=0x%08x, wr_spaddr=0x%08x, iter=%d\n",
          op_spaddr, wr_spaddr, iter);

  // Transpose matrix: input is DIM x iter, output is iter x DIM
  // iter is the depth dimension, maximum supported is 16x16
  size_t depth = iter;
  assert(depth <= 16 && "Transpose depth must be less than or equal to 16");

  // Transpose DIM x iter matrix to iter x DIM matrix
  for (size_t i = 0; i < DIM; i++) {
    for (size_t j = 0; j < depth; j++) {
      // Transpose: result[j][i] = input[i][j]
      uint32_t op_bank = spad_get_bank(op_spaddr + i);
      uint32_t op_offset = spad_get_offset(op_spaddr + i);
      uint32_t wr_bank = spad_get_bank(wr_spaddr + j);
      uint32_t wr_offset = spad_get_offset(wr_spaddr + j);

      elem_t val = toy_state.banks.at(op_bank).at(op_offset).at(j);
      toy_state.banks.at(wr_bank).at(wr_offset).at(i) = val;
    }
  }
}
