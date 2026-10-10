// RUN: buddy-opt %s --target=toy -assign-physical-banks -split-input-file -verify-diagnostics

func.func @captured_allocation(%condition: i1) {
  %source = buckyball.bank_alloc
  %target = arith.constant 2 : i64
  scf.if %condition {
    // expected-error @+1 {{bank transfer cannot consume a control-flow bank alias}}
    %aggregate = buckyball.bank_transfer %source %target : i64 i64
    buckyball.bank_release %aggregate : i64
  }
  return
}
// -----
func.func @if_result_alias(%condition: i1) {
  %source = buckyball.bank_alloc
  %target = arith.constant 2 : i64
  %alias = scf.if %condition -> i64 {
    scf.yield %source : i64
  } else {
    scf.yield %source : i64
  }
  // expected-error @+1 {{bank transfer cannot consume a control-flow bank alias}}
  %aggregate = buckyball.bank_transfer %alias %target : i64 i64
  buckyball.bank_release %aggregate : i64
  return
}
// -----
func.func @loop_result_alias() {
  %source = buckyball.bank_alloc
  %target = arith.constant 2 : i64
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %alias = scf.for %i = %zero to %one step %one iter_args(%bank = %source) -> i64 {
    scf.yield %bank : i64
  }
  // expected-error @+1 {{bank transfer cannot consume a control-flow bank alias}}
  %aggregate = buckyball.bank_transfer %alias %target : i64 i64
  buckyball.bank_release %aggregate : i64
  return
}
