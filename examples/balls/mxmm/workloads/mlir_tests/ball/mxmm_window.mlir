// RUN: buddy-opt %s --target=toy -assign-physical-banks -lower-bank-ssa-to-intrinsics -lower-buckyball -canonicalize | FileCheck %s
// CHECK-LABEL: func.func @window
// CHECK: buckyball.intr.custom {{.*}} <funct7 = 75>
// CHECK-LABEL: func.func @window_small
// CHECK-DAG: %[[RS1:[^ ]+]] = arith.constant 34367084547 : i64
// CHECK-DAG: %[[RS2:[^ ]+]] = arith.constant 9007474250153985 : i64
// CHECK: buckyball.intr.custom %[[RS1]], %[[RS2]] <funct7 = 75>
func.func @window(%a: i64, %b: i64, %c: i64, %rows: i64, %cols: i64, %count: i64, %full: i64, %start: i64, %first: i1, %last: i1, %base: i64) attributes {llvm.emit_c_interface} {
  buckyball.mxfp8_window %a, %b, %c, %rows, %cols, %count, %full, %start, %first, %last, %base : i64
  return
}
func.func @window_small() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64
  %b = arith.constant 6 : i64
  %c = arith.constant 7 : i64
  %rows = arith.constant 1 : i64
  %cols = arith.constant 16 : i64
  %count = arith.constant 32 : i64
  %full = arith.constant 64 : i64
  %start = arith.constant 32 : i64
  %first = arith.constant true
  %last = arith.constant true
  %base = arith.constant 1 : i64
  buckyball.mxfp8_window %a, %b, %c, %rows, %cols, %count, %full, %start, %first, %last, %base : i64
  return
}
