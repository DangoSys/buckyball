// RUN: buddy-opt %s -lower-buckyball -canonicalize | FileCheck %s
// CHECK-LABEL: func.func @first
// CHECK-DAG: arith.constant 34362886145 : i64
// CHECK-DAG: arith.constant 83951632 : i64
// CHECK-LABEL: func.func @last
// CHECK-DAG: arith.constant 34362886145 : i64
// CHECK-DAG: arith.constant 100728848 : i64
// The caller loads A into bank 1, B into bank 2 and C into bank 3. Two steps of
// one chain over the same A and B: the first only accumulates, the last
// accumulates and writes the 16x16 sum to C from row 1.
func.func @first() attributes {llvm.emit_c_interface} {
  mxmm_ball.matmul_mxfp8 <{a = 1 : i64, b = 2 : i64, c = 3 : i64, rows = 16 : i64, cols = 16 : i64, k = 32 : i64, base = 1 : i64, first}>
  return
}
func.func @last() attributes {llvm.emit_c_interface} {
  mxmm_ball.matmul_mxfp8 <{a = 1 : i64, b = 2 : i64, c = 3 : i64, rows = 16 : i64, cols = 16 : i64, k = 32 : i64, base = 1 : i64, last}>
  return
}
