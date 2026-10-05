// RUN: buddy-opt %s --target=toy -assign-physical-banks -lower-bank-ssa-to-intrinsics -lower-buckyball -canonicalize | FileCheck %s
// CHECK-LABEL: func.func @quant_block_32
// CHECK: %[[ZERO:[^ ]+]] = arith.constant 0 : i64
// CHECK: %[[PACKED:[^ ]+]] = arith.constant 34363932675 : i64
// CHECK: buckyball.intr.custom %[[PACKED]], %[[ZERO]] {funct7 = 54 : i32}
// CHECK-LABEL: func.func @quant_dynamic
// CHECK: buckyball.intr.custom {{.*}} {funct7 = 54 : i32}
func.func @quant_block_32() attributes {llvm.emit_c_interface} {
  %input = arith.constant 3 : i64
  %output = arith.constant 4 : i64
  %count = arith.constant 32 : i64
  buckyball.mxquant %input, %output, %count : i64
  return
}
func.func @quant_dynamic(%count: i64) attributes {llvm.emit_c_interface} {
  %input = arith.constant 3 : i64
  %output = arith.constant 4 : i64
  buckyball.mxquant %input, %output, %count : i64
  return
}
