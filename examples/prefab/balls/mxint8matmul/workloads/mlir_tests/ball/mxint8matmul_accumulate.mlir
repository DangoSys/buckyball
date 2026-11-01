// RUN: buddy-opt %s -lower-acc -canonicalize | FileCheck %s
// CHECK-LABEL: func.func @accumulate
// CHECK-DAG: arith.constant 103619233793 : i64
// CHECK-DAG: arith.constant 4611686018427400197 : i64
// The caller loads A into bank 1, B into bank 2 and the old C into bank 3,
// each from bank row 0.
func.func @accumulate() attributes {llvm.emit_c_interface} {
  mxint8matmul_ball.matmul_mxint8 <{a = 1 : i64, b = 2 : i64, c = 3 : i64, m = 5 : i64, n = 12 : i64, k = 96 : i64, accumulate = true}>
  return
}
