// The banks 1 (A), 2 (B) and 3 (C) are allocated and loaded by the caller.
func.func @redmule_chain() attributes {llvm.emit_c_interface} {
  buckyball.custom "matmul_f16" <{sources = array<i64: 1, 2>, destination = 3 : i64}> {rows = 2 : i64, k = 8 : i64}
  buckyball.custom "matmul_f16" <{sources = array<i64: 1, 2>, destination = 3 : i64, read_destination}> {rows = 2 : i64, k = 8 : i64}
  return
}
