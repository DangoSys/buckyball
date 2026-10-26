// RUN: buddy-opt %s -assign-physical-banks -lower-buckyball -canonicalize | FileCheck %s
// CHECK-LABEL: func.func @quant_block_32
// CHECK-DAG: %[[F54:.*]] = llvm.mlir.constant(54 : i32) : i32
// CHECK: %[[ZERO:[^ ]+]] = arith.constant 0 : i64
// CHECK: %[[PACKED:[^ ]+]] = arith.constant 34363932675 : i64
// CHECK: llvm.call_intrinsic "llvm.riscv.buckyball.custom"(%[[PACKED]], %[[ZERO]], %[[F54]])
// CHECK-LABEL: func.func @quant_dynamic
// CHECK-DAG: %[[F54:.*]] = llvm.mlir.constant(54 : i32) : i32
// CHECK: llvm.call_intrinsic "llvm.riscv.buckyball.custom"({{.*}}, %[[F54]])
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
