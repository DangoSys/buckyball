// RUN: buddy-opt %s -lower-buckyball -canonicalize | FileCheck %s
// CHECK-LABEL: func.func @quant
// CHECK-DAG: arith.constant 68721573889 : i64
// CHECK-DAG: arith.constant 0 : i64
// The caller loads 64 FP32 into bank 1 and the output guard into bank 2.
func.func @quant() attributes {llvm.emit_c_interface} {
  mxquant_ball.quant_mxfp8 <{input = 1 : i64, output = 2 : i64, count = 64 : i64}>
  return
}
