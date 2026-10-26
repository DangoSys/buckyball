// RUN: buddy-opt %s -split-input-file -lower-buckyball -verify-diagnostics
func.func @zero_rows() {
  // expected-error @+2 {{add_f32 requires distinct banks, rows within a bank and a group in [0, 31]}}
  // expected-error @+1 {{has no lowering on this core}}
  buckyball.custom "add_f32" <{sources = array<i64: 32, 33>, destination = 34 : i64}> {rows = 0 : i64, group = 0 : i64, first = true}
  return
}
// -----
func.func @wide_group() {
  // expected-error @+2 {{add_f32 requires distinct banks, rows within a bank and a group in [0, 31]}}
  // expected-error @+1 {{has no lowering on this core}}
  buckyball.custom "add_f32" <{sources = array<i64: 32, 33>, destination = 34 : i64}> {rows = 1 : i64, group = 32 : i64, first = false}
  return
}
// -----
func.func @wide_rows() {
  // expected-error @+2 {{add_f32 requires distinct banks, rows within a bank and a group in [0, 31]}}
  // expected-error @+1 {{has no lowering on this core}}
  buckyball.custom "add_f32" <{sources = array<i64: 32, 33>, destination = 34 : i64}> {rows = 17179869184 : i64, group = 0 : i64, first = false}
  return
}
// -----
func.func @equal_banks() {
  // expected-error @+2 {{add_f32 requires distinct banks, rows within a bank and a group in [0, 31]}}
  // expected-error @+1 {{has no lowering on this core}}
  buckyball.custom "add_f32" <{sources = array<i64: 32, 33>, destination = 32 : i64}> {rows = 1 : i64, group = 0 : i64, first = true}
  return
}
