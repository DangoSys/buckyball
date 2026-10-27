// The banks 1 (A), 2 (B) and 3 (C) are allocated and loaded by the caller.
func.func @redmule_chain() attributes {llvm.emit_c_interface} {
  redmule_ball.matmul_f16 <{a = 1 : i64, b = 2 : i64, c = 3 : i64, rows = 2 : i64, k = 8 : i64}>
  redmule_ball.matmul_f16 <{a = 1 : i64, b = 2 : i64, c = 3 : i64, rows = 2 : i64, k = 8 : i64, read_destination}>
  return
}
