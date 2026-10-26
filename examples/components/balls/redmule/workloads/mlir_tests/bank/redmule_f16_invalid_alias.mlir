// RUN: buddy-opt %s -assign-physical-banks -lower-buckyball -verify-diagnostics
func.func @invalid_alias() {
  %a = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %b = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  // expected-error @+1 {{FP16 matmul requires distinct A, B and C banks}}
  %c = buckyball.bank_custom "matmul_f16" (%a, %b) -> %a {rows = 1 : i64, k = 8 : i64}
  buckyball.bank_release %c
  buckyball.bank_release %b
  return
}
