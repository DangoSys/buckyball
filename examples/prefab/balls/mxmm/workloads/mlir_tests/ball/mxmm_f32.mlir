// RUN: buddy-opt %s -lower-buckyball -canonicalize | FileCheck %s
// CHECK-LABEL: func.func @single
// CHECK-DAG: arith.constant 4298115073 : i64
// CHECK-DAG: arith.constant 50397200 : i64
// The caller loads A into bank 1, B into bank 2 and C into bank 3. One step
// that both starts and ends its chain.
func.func @single() attributes {llvm.emit_c_interface} {
  mxmm_ball.matmul_f32 <{a = 1 : i64, b = 2 : i64, c = 3 : i64, rows = 16 : i64, cols = 16 : i64, k = 4 : i64, base = 0 : i64, first, last}>
  return
}
