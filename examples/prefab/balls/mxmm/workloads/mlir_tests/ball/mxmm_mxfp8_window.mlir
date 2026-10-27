// RUN: buddy-opt %s -lower-buckyball -canonicalize | FileCheck %s
// CHECK-LABEL: func.func @window
// CHECK-DAG: arith.constant 34362886145 : i64
// CHECK-DAG: arith.constant 9007474183045121 : i64
// The caller loads A into bank 1, B into bank 2 and C into bank 3. One step
// reads columns 32..63 of a one-row A of 64 columns.
func.func @window() attributes {llvm.emit_c_interface} {
  mxmm_ball.matmul_mxfp8_window <{a = 1 : i64, b = 2 : i64, c = 3 : i64, rows = 1 : i64, cols = 16 : i64, k = 32 : i64, full_k = 64 : i64, start_k = 32 : i64, base = 0 : i64, first, last}>
  return
}
