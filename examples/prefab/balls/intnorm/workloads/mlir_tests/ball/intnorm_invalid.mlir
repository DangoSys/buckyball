// RUN: buddy-opt %s -split-input-file -lower-acc -verify-diagnostics -o /dev/null
func.func @scratch_exceeds_bank() {
  // expected-error @+1 {{integer normalization exceeds the bank or ISA fields}}
  intnorm_ball.norm_mxint8 <{input = 0 : i64, coefficients = 1 : i64,
    output = 2 : i64, rows = 1 : i64, k = 512 : i64,
    in_base = 0 : i64, coefficient_base = 0 : i64, out_base = 1000 : i64}>
  return
}
// -----
func.func @unaligned_width() {
  // expected-error @+1 {{integer normalization exceeds the bank or ISA fields}}
  intnorm_ball.norm_mxint8 <{input = 0 : i64, coefficients = 1 : i64,
    output = 2 : i64, rows = 1 : i64, k = 32 : i64,
    in_base = 0 : i64, coefficient_base = 0 : i64, out_base = 0 : i64}>
  return
}
// -----
func.func @aliased_banks() {
  // expected-error @+1 {{integer normalization requires three distinct banks}}
  intnorm_ball.norm_mxint8 <{input = 0 : i64, coefficients = 1 : i64,
    output = 0 : i64, rows = 1 : i64, k = 512 : i64,
    in_base = 0 : i64, coefficient_base = 0 : i64, out_base = 0 : i64}>
  return
}
