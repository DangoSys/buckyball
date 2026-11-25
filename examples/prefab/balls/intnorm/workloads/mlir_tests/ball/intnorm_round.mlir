// RUN: buddy-opt %s -lower-acc -canonicalize | FileCheck %s
// CHECK: llvm.mlir.constant(77 : i32)
// CHECK: llvm.call_intrinsic "llvm.riscv.buckyball.custom"
func.func @normalize() attributes {llvm.emit_c_interface} {
  intnorm_ball.norm_mxint8 <{input = 1 : i64, coefficients = 2 : i64,
    output = 3 : i64, rows = 1 : i64, k = 512 : i64,
    in_base = 0 : i64, coefficient_base = 0 : i64, out_base = 0 : i64}>
  return
}
