// RUN: buddy-opt %s -lower-acc -canonicalize | FileCheck %s
// CHECK-LABEL: func.func @quant
// CHECK-DAG: arith.constant 3223322625 : i64
// CHECK-DAG: arith.constant 544 : i64
// The caller loads 3 rows of 544 FP32 into bank 1; bank 2 receives the
// packed rows from bank row 0.
func.func @quant() attributes {llvm.emit_c_interface} {
  mxint8quantize_ball.quant_mxint8 <{input = 1 : i64, output = 2 : i64, rows = 3 : i64, k = 544 : i64}>
  return
}
