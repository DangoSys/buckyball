// RUN: buddy-opt %s -assign-physical-banks -lower-acc -canonicalize | FileCheck %s
// CHECK: llvm.mlir.constant(77 : i32)
// CHECK: llvm.call_intrinsic "llvm.riscv.buckyball.custom"
func.func @normalize(%input: memref<128x4xf32>,
                     %coefficients: memref<257x4xi32>,
                     %output: memref<33x16xi8>) attributes {llvm.emit_c_interface} {
  %a = bbacc.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %b = bbacc.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %c = bbacc.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %x = bbacc.bank_mvin %input, %a <{depth = 128 : i64, stride = 1 : i64}> : memref<128x4xf32>
  %p = bbacc.bank_mvin %coefficients, %b <{depth = 257 : i64, stride = 1 : i64}> : memref<257x4xi32>
  %y = intnorm_ball.bank_norm_mxint8 %x, %p -> %c <{rows = 1 : i64, k = 512 : i64,
    in_base = 0 : i64, coefficient_base = 0 : i64, out_base = 0 : i64}>
  %stored = bbacc.bank_mvout %output, %y <{depth = 33 : i64, stride = 1 : i64}> : memref<33x16xi8>
  bbacc.bank_release %x
  bbacc.bank_release %p
  bbacc.bank_release %stored
  return
}
