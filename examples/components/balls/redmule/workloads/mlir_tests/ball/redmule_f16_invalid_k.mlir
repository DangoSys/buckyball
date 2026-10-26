// RUN: buddy-opt %s -lower-buckyball -verify-diagnostics
func.func @invalid_k() {
  // expected-error @+2 {{FP16 matmul shape exceeds the bank or ISA geometry}}
  // expected-error @+1 {{has no lowering on this core}}
  buckyball.custom "matmul_f16" <{sources = array<i64: 0, 1>, destination = 2 : i64}> {rows = 1 : i64, k = 7 : i64, op1_base = 0 : i64, op2_base = 0 : i64, wr_base = 0 : i64}
  return
}
