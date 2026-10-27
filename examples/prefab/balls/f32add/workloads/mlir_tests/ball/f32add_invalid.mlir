// RUN: buddy-opt %s -split-input-file -lower-buckyball -verify-diagnostics
func.func @zero_rows() {
  // expected-error @+1 {{add_f32 requires distinct A, B and C banks and rows within a bank}}
  f32add_ball.add_f32 <{a = 32 : i64, b = 33 : i64, c = 34 : i64, rows = 0 : i64}>
  return
}
// -----
func.func @wide_rows() {
  // expected-error @+1 {{add_f32 requires distinct A, B and C banks and rows within a bank}}
  f32add_ball.add_f32 <{a = 32 : i64, b = 33 : i64, c = 34 : i64, rows = 17179869184 : i64}>
  return
}
// -----
func.func @equal_banks() {
  // expected-error @+1 {{add_f32 requires distinct A, B and C banks and rows within a bank}}
  f32add_ball.add_f32 <{a = 32 : i64, b = 33 : i64, c = 32 : i64, rows = 1 : i64}>
  return
}
