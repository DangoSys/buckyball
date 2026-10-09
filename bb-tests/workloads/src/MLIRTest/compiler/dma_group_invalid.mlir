// RUN: buddy-opt %s -split-input-file -verify-diagnostics
func.func @negative(%input: memref<16x16xi8>, %bank: i64, %depth: i64, %stride: i64) {
  // expected-error @+1 {{group must be in [0, 31]}}
  buckyball.mvin %input %bank %depth %stride <group = -1> : memref<16x16xi8> i64 i64 i64
  return
}

// -----

func.func @overflow(%output: memref<16x16xi8>, %bank: i64, %depth: i64, %stride: i64) {
  // expected-error @+1 {{group must be in [0, 31]}}
  buckyball.mvout %output %bank %depth %stride <group = 32> : memref<16x16xi8> i64 i64 i64
  return
}
