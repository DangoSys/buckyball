// RUN: buddy-opt %s -split-input-file -verify-diagnostics
func.func @negative() {
  %source = arith.constant -1 : i64
  %target = arith.constant 17 : i64
  // expected-error @+1 {{bank ID must be in [0, 1023]}}
  buckyball.mset_transfer %source %target : i64 i64
  return
}
// -----
func.func @overflow() {
  %source = arith.constant 1024 : i64
  %target = arith.constant 17 : i64
  // expected-error @+1 {{bank ID must be in [0, 1023]}}
  buckyball.mset_transfer %source %target : i64 i64
  return
}
// -----
func.func @self() {
  %source = arith.constant 17 : i64
  // expected-error @+1 {{source and target must differ}}
  buckyball.mset_transfer %source %source : i64 i64
  return
}
