// RUN: buddy-opt %s -lower-buckyball -verify-diagnostics
func.func @invalid_k() {
  // expected-error @+1 {{FP16 matmul shape exceeds the bank or ISA geometry}}
  buckyball.custom "matmul_f16" <{sources = array<i64: 1, 2>, destination = 3 : i64}> {rows = 1 : i64, k = 7 : i64}
  return
}
