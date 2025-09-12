#include "common.h"
#include "inst.h"
#include "toy.h"

// Sparse matrix multiplication
// rs1: A_addr[spAddrLen-1:0] | B_addr[2*spAddrLen-1:spAddrLen]
// rs2: row_rf_bank[0] | col_rf_bank[1] | C_addr[spAddrLen+1:2] | nnz[27:16]
void toy_t::sparse_mul(reg_t rs1, reg_t rs2) {
  sparse_mul_rs1_t rs1_fields(rs1);
  sparse_mul_rs2_t rs2_fields(rs2);

  auto const A_addr = rs1_fields.A_addr();
  auto const B_addr = rs1_fields.B_addr();
  auto const row_rf_bank = rs2_fields.row_rf_bank();
  auto const col_rf_bank = rs2_fields.col_rf_bank();
  auto const C_addr = rs2_fields.C_addr();
  auto const nnz = rs2_fields.nnz();

  dprintf(p, "TOY: sparse_mul - rs1=0x%08lx, rs2=0x%08lx\n", rs1, rs2);
  dprintf(p,
          "TOY: sparse_mul - A_addr=0x%08x, B_addr=0x%08x, C_addr=0x%08x, "
          "row_rf=%d, col_rf=%d, nnz=%d\n",
          A_addr, B_addr, C_addr, row_rf_bank, col_rf_bank, nnz);

  // Note: Do NOT initialize C to zero here - this function performs C += A * B
  // The caller is responsible for clearing C if needed

  // Get register file data
  auto &row_indices = toy_state.rf.at(row_rf_bank);
  auto &col_ptrs = toy_state.rf.at(col_rf_bank);

  // Process CSC sparse matrix multiplication: C = A_sparse * B
  // CSC format: values are stored row by row, column pointers indicate start of
  // each column
  for (size_t col = 0; col < DIM; col++) {
    int col_start = col_ptrs.at(col);
    int col_end = col_ptrs.at(col + 1);

    for (int nz_idx = col_start; nz_idx < col_end && nz_idx < nnz; nz_idx++) {
      int row = row_indices.at(nz_idx);

      if (row >= 0 && row < DIM && col >= 0 && col < DIM) {
        // Access sparse A value from scratchpad
        // Values are stored linearly in scratchpad, packed by rows
        // nz_idx corresponds to the position in the values array
        size_t val_row = A_addr + (nz_idx / DIM);
        size_t val_col = nz_idx % DIM;
        elem_t a_val = toy_state.spad.at(val_row).at(val_col);

        // Perform: C[row, :] += a_val * B[col, :]
        for (size_t j = 0; j < DIM; j++) {
          toy_state.spad.at(C_addr + row).at(j) +=
              a_val * toy_state.spad.at(B_addr + col).at(j);
        }
      }
    }
  }
}
