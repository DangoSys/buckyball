// RUN: buddy-opt %s --target=main -split-input-file -verify-diagnostics
func.func @zero_rows(%a: i64, %b: i64, %c: i64) {
  // expected-error @+1 {{rows must be in [1, 17179869183]}}
  buckyball.f32add %a, %b, %c <rows = 0, group = 0, first = true> : i64
  return
}
// -----
func.func @wide_group(%a: i64, %b: i64, %c: i64) {
  // expected-error @+1 {{group must be in [0, 31]}}
  buckyball.f32add %a, %b, %c <rows = 1, group = 32, first = false> : i64
  return
}
// -----
func.func @wide_rows(%a: i64, %b: i64, %c: i64) {
  // expected-error @+1 {{rows must be in [1, 17179869183]}}
  buckyball.f32add %a, %b, %c <rows = 17179869184, group = 0, first = false> : i64
  return
}
// -----
func.func @equal_bank_constants() {
  %a = arith.constant 32 : i64
  %b = arith.constant 33 : i64
  %c = arith.constant 32 : i64
  // expected-error @+1 {{source, accumulator and output banks must differ}}
  buckyball.f32add %a, %b, %c <rows = 1, group = 0, first = true> : i64
  return
}
// -----
func.func @wide_bank() {
  %a = arith.constant 1024 : i64
  %b = arith.constant 33 : i64
  %c = arith.constant 34 : i64
  // expected-error @+1 {{bank ID must be in [0, 1023]}}
  buckyball.f32add %a, %b, %c <rows = 1, group = 0, first = false> : i64
  return
}
