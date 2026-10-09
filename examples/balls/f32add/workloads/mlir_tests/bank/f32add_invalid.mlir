// RUN: buddy-opt %s --target=main -split-input-file -verify-diagnostics
func.func @same_ssa_bank(%a: i64, %b: i64) {
  // expected-error @+1 {{source, accumulator and output banks must differ}}
  %out = buckyball.bank_f32add %a %b %b <rows = 1, group = 0, first = true> : i64 i64 i64
  return
}
// -----
func.func @zero_rows(%a: i64, %b: i64, %c: i64) {
  // expected-error @+1 {{rows must be in [1, 17179869183]}}
  %out = buckyball.bank_f32add %a %b %c <rows = 0, group = 0, first = true> : i64 i64 i64
  return
}
// -----
func.func @negative_group(%a: i64, %b: i64, %c: i64) {
  // expected-error @+1 {{group must be in [0, 31]}}
  %out = buckyball.bank_f32add %a %b %c <rows = 1, group = -1, first = false> : i64 i64 i64
  return
}
// -----
func.func @negative_bank() {
  %a = arith.constant -1 : i64
  %b = arith.constant 33 : i64
  %c = arith.constant 34 : i64
  // expected-error @+1 {{bank ID must be in [0, 1023]}}
  %out = buckyball.bank_f32add %a %b %c <rows = 1, group = 0, first = true> : i64 i64 i64
  return
}
