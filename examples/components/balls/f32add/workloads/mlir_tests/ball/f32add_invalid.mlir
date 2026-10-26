// RUN: buddy-opt %s -split-input-file -lower-buckyball -verify-diagnostics
func.func @zero_rows() {
  // expected-error @+1 {{add_f32 requires distinct A, B and C banks, rows within a bank and no read_destination}}
  buckyball.custom "add_f32" <{sources = array<i64: 32, 33>, destination = 34 : i64}> {rows = 0 : i64}
  return
}
// -----
func.func @wide_rows() {
  // expected-error @+1 {{add_f32 requires distinct A, B and C banks, rows within a bank and no read_destination}}
  buckyball.custom "add_f32" <{sources = array<i64: 32, 33>, destination = 34 : i64}> {rows = 17179869184 : i64}
  return
}
// -----
func.func @equal_banks() {
  // expected-error @+1 {{add_f32 requires distinct A, B and C banks, rows within a bank and no read_destination}}
  buckyball.custom "add_f32" <{sources = array<i64: 32, 33>, destination = 32 : i64}> {rows = 1 : i64}
  return
}
// -----
func.func @read_destination() {
  // expected-error @+1 {{add_f32 requires distinct A, B and C banks, rows within a bank and no read_destination}}
  buckyball.custom "add_f32" <{sources = array<i64: 32, 33>, destination = 34 : i64, read_destination}> {rows = 1 : i64}
  return
}
