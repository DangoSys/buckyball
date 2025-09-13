#include "common.h"
#include "inst.h"
#include "toy.h"

// Im2col operation for convolution
// rs1: op_spaddr[13:0] | wr_spaddr[27:14]
// rs2: kcol[23] | krow[27] | inrow[31:45] | incol[36:47] | startrow[46:57] |
// startcol[51:62]
void toy_t::im2col(reg_t rs1, reg_t rs2) {
  im2col_rs1_t rs1_fields(rs1);
  im2col_rs2_t rs2_fields(rs2);

  auto const op_spaddr = rs1_fields.op_spaddr();
  auto const wr_spaddr = rs1_fields.wr_spaddr();
  auto const kcol = rs2_fields.kcol();
  auto const krow = rs2_fields.krow();
  auto const inrow = rs2_fields.inrow();
  auto const incol = rs2_fields.incol();
  auto const startrow = rs2_fields.startrow();
  auto const startcol = rs2_fields.startcol();

  dprintf(p, "TOY: im2col - rs1=0x%08lx, rs2=0x%08lx\n", rs1, rs2);
  dprintf(p,
          "TOY: im2col - op_spaddr=0x%08x, wr_spaddr=0x%08x, kcol=%d, krow=%d, "
          "inrow=%d, incol=%d, startrow=%d, startcol=%d\n",
          op_spaddr, wr_spaddr, kcol, krow, inrow, incol, startrow, startcol);

  // Im2col transforms input image patches into columns for matrix
  // multiplication This is commonly used in convolution operations

  // Calculate output dimensions
  size_t out_row = 0;

  // Iterate through kernel positions
  for (size_t kr = 0; kr < krow; kr++) {
    for (size_t kc = 0; kc < kcol; kc++) {
      size_t out_col = 0;

      // Iterate through output positions
      for (size_t sr = startrow; sr < startrow + inrow && sr + kr < inrow;
           sr++) {
        for (size_t sc = startcol; sc < startcol + incol && sc + kc < incol;
             sc++) {
          // Calculate input position
          size_t in_r = sr + kr;
          size_t in_c = sc + kc;

          // Bounds checking
          if (in_r < inrow && in_c < incol && out_row < DIM && out_col < DIM) {
            // Copy from input image to im2col matrix
            uint32_t op_bank = spad_get_bank(op_spaddr + in_r);
            uint32_t op_offset = spad_get_offset(op_spaddr + in_r);
            uint32_t wr_bank = spad_get_bank(wr_spaddr + out_row);
            uint32_t wr_offset = spad_get_offset(wr_spaddr + out_row);

            elem_t val = toy_state.banks.at(op_bank).at(op_offset).at(in_c);
            toy_state.banks.at(wr_bank).at(wr_offset).at(out_col) = val;
          }

          out_col++;
          if (out_col >= DIM)
            break;
        }
        if (out_col >= DIM)
          break;
      }

      out_row++;
      if (out_row >= DIM)
        break;
    }
    if (out_row >= DIM)
      break;
  }
}
