// The banks 0 (A), 1 (B) and 2 (C) are allocated and loaded by the caller.
func.func @redmule_chain() attributes {llvm.emit_c_interface} {
  buckyball.custom "matmul_f16" <{sources = array<i64: 0, 1>, destination = 2 : i64}> {rows = 2 : i64, k = 8 : i64, op1_base = 0 : i64, op2_base = 0 : i64, wr_base = 0 : i64}
  buckyball.custom "matmul_f16" <{sources = array<i64: 0, 1>, destination = 2 : i64, read_destination}> {rows = 2 : i64, k = 8 : i64, op1_base = 0 : i64, op2_base = 0 : i64, wr_base = 0 : i64}
  return
}
