// RUN: buddy-opt %s -lower-buckyball -verify-diagnostics
func.func @invalid_k() {
  // expected-error @+1 {{FP16 matmul shape exceeds the bank or ISA geometry}}
  redmule_ball.matmul_f16 <{a = 1 : i64, b = 2 : i64, c = 3 : i64, rows = 1 : i64, k = 7 : i64}>
  return
}
