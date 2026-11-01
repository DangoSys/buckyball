// RUN: buddy-opt %s -assign-physical-banks -lower-acc -verify-diagnostics
func.func @invalid_alias() {
  %a = bbacc.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %b = bbacc.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  // expected-error @+1 {{FP16 matmul requires distinct A, B and C banks}}
  %c = redmule_ball.bank_matmul_f16 %a, %b -> %a <{rows = 1 : i64, k = 8 : i64}>
  bbacc.bank_release %c
  bbacc.bank_release %b
  return
}
